import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, StochasticWeightAveraging, LearningRateMonitor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import wandb
import os
import random
import timm
from einops import repeat
import pandas as pd
from moca import PModel
# 设置种子以确保结果的可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(202406)
wandb.init(project="HiCur-NPC", name="Pro")
learning_rate = 1e-5
num_epochs = 20
image_size = 384
valid_trans = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])
timm_trans = A.Compose([
    A.Resize(256,256),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])


def pad_or_truncate(tensor, max_length=32,pad_size=256):
    current_length = tensor.shape[0]
    if current_length > max_length:
        # 截断到 max_length
        return tensor[:max_length]
    else:
        # 补零到 max_length
        padding = torch.zeros((max_length - current_length, 3,pad_size,pad_size), device=tensor.device)
        return torch.cat((tensor, padding), dim=0)
class TyDataset(Dataset):
    def __init__(self,ty_js):
        self.ty_js = ty_js
    
    def __getitem__(self, idx):
        row = self.ty_js[idx]
        x= {}
        image_file = './data/pro/imgs/' + row['image']
        imgs = []
        imgs_timm = []
        files = os.listdir(image_file)
        for file in files:
            img = cv2.imread(image_file+'/'+file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = valid_trans(image = img)
            image_tensor = aug['image']
            imgs.append(image_tensor)
            aug1 = timm_trans(image = img)
            image_tensor1 = aug1['image']
            imgs_timm.append(image_tensor1)
        x['imgs'] = pad_or_truncate(torch.stack(imgs),pad_size=384)
        x['imgs_timm'] = pad_or_truncate(torch.stack(imgs_timm))
        x['image_size'] = img.size
        prompt = row['text']
        x['prompt'] = prompt
        x['pt'] = torch.load('./data/pro-images-cga-pt/'+row['image']+'.pt',map_location=torch.device('cpu')).squeeze(dim=0)
        y = torch.tensor([int(row['survival']),int(row['metastasis'])])
        return x,y
    def __len__(self):
        return len(self.ty_js)

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = PModel()
        self.criterion = nn.SmoothL1Loss()
        train_js = []
        val_js = []
        with open('./data/pro/train_pro.json', 'r') as f:
            for line in f:
                data = json.loads(line)
                train_js.append(data)
        with open('./data/pro/val_pro.json', 'r') as f:
            for line in f:
                data = json.loads(line)
                val_js.append(data)
        train_data = TyDataset(train_js)
        val_data = TyDataset(val_js)
        self.dl_train = DataLoader(train_data, batch_size=8, num_workers = 4, shuffle = True,pin_memory = True)
        self.dl_val = DataLoader(val_data, batch_size=8, num_workers = 4, shuffle = False,pin_memory = True)

    
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        return self.dl_train
    def val_dataloader(self):
        return [self.dl_train, self.dl_val]
    def training_step(self, batch, batch_idx):
        x,y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y.float())
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx, dataloader_idx):
        x,y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y.float())
        self.log("val_loss", loss)
        return y, yhat
    def validation_step_end(self, outputs):
        return outputs
    def validation_epoch_end(self, outputs_list):
        def roc_auc_score_uni(y_true, y_score):
            try:
                return roc_auc_score(y_true, y_score)
            except ValueError:  
                return 0.0

        for dataloader_idx, outputs in enumerate(outputs_list):
            y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
            logits = torch.cat([_[1] for _ in outputs]).detach().cpu().numpy()
            
            num_labels = y.shape[1]
            aucs = []
            for i in range(num_labels):
                auc = roc_auc_score_uni(y[:, i], logits[:, i])
                aucs.append(auc)
                label_type = "train" if dataloader_idx == 0 else "val"
                self.log(f"{label_type}_auc_label_{i}", auc, prog_bar=True)
            average_auc = np.mean(aucs)
            self.log(f"{label_type}_average_auc", average_auc, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.parameters()},
        ], lr=learning_rate)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=20,
                eta_min=0
            ),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    
trainer = pl.Trainer(
    gpus=1,                 # Number of GPUs to use
    precision=16,           # Use mixed precision (FP16)
    max_epochs=20,           # Number of epochs
    gradient_clip_val=0.5,  # Gradient clipping to prevent exploding gradients
    enable_checkpointing=False,
    logger=WandbLogger(project="HiCur-NPC", name="Prognosis"),  # TensorBoard logger
    callbacks=[          
        ModelCheckpoint(
            dirpath="./model/pro/",
            filename="hicur_pro",
            monitor="val_loss",
            mode="min"
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(leave = True)
    ]
)

model = Model()
trainer.fit(model)