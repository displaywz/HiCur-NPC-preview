import pytorch_lightning as pl
import torch
import torch.nn as nn
import cv2
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from metrices import iou,dice
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar,LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from math import sqrt
from moeblock import MixtralSparseMoeBlock
from moca import ConvNeXtUNet
from seg_loss import FocalDiceLoss
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2024)  # 设置任意种子值
wandb.init(project="HiCur-NPC", name="Segment")
image_size = 256
valid_trans = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean = [0.53017267, 0.31335013, 0.31556818], std = [0.33659712, 0.23721003, 0.24286168]),
    ToTensorV2()
])
    
class TyDataset(Dataset):
    def __init__(self,ty_js):
        self.ty_js = ty_js
    
    def __getitem__(self, idx):
        row = self.ty_js.iloc[idx]
        x= {}
        image_file = './data/seg/imgs/' + row['image']
        mask_file = './data/seg/masks/' + row['mask']
        pt_file = './data/seg-images-cga-pt/' + row['image'].split('/')[-1].replace('.png','.pt')
        llava_pt = torch.load(pt_file,map_location=torch.device('cpu')).squeeze(dim=0)
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = valid_trans(image = img)
        image_tensor = aug['image']
        x['img'] = image_tensor
        x['image_size'] = img.size
        x['text'] = llava_pt
        label = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (256, 256))
        label = np.where(label > 10, 1, 0)
        y = torch.tensor(label).unsqueeze(dim=0)
        return x,y
    def __len__(self):
        return len(self.ty_js)


    
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = ConvNeXtUNet()
        self.criterion = FocalDiceLoss()
        with open('./data/seg/image2label_train.json', 'r') as file:
            data = json.load(file)
        data_list = []
        for img, masks in data.items():
            for mask in masks:
                data_list.append({'image': img, 'mask': mask})
        train_df = pd.DataFrame(data_list)
        with open('./data/seg/image2label_val.json', 'r') as file:
            data = json.load(file)
        data_list = []
        for img, masks in data.items():
            for mask in masks:
                data_list.append({'image': img, 'mask': mask})
        val_df = pd.DataFrame(data_list)
        train_data = TyDataset(train_df)
        self.dl_train = DataLoader(train_data, batch_size=16, num_workers = 8, pin_memory = True, shuffle = True)
        val_data = TyDataset(val_df)
        self.dl_val = DataLoader(val_data, batch_size=16, num_workers = 8, pin_memory = True, shuffle = False)
    def forward(self, x):
        return self.model(x['img'],x['text'])
    def train_dataloader(self):
        return self.dl_train
    def val_dataloader(self):
        return self.dl_val
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =  self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return {"y_true": y, "y_pred": y_hat}
    def validation_epoch_end(self, outputs):
        y_trues = torch.cat([output["y_true"] for output in outputs])
        y_preds = torch.cat([output["y_pred"] for output in outputs])
        mIOU = np.mean(iou(y_trues, y_preds))
        mDice = np.mean(dice(y_trues,y_preds))
        print('miou:',mIOU)
        print('mdice:',mDice)
        self.log("mIOU", mIOU)
        self.log("mDice", mDice)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =  self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss, "y_true": y, "y_pred": y_hat}
    def configure_optimizers(self):
        learning_rate = 1e-4  
        weight_decay = 0.001  
        optimizer = torch.optim.AdamW(
            [
                {"params":self.model.parameters()},
            ],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr = learning_rate, 
                steps_per_epoch = int(len(self.train_dataloader())), 
                epochs = 20, 
                anneal_strategy = "cos", 
                final_div_factor = 30,), 
            'name': 'learning_rate', 
            'interval':'step', 
            'frequency': 1
        }
        return [optimizer], [scheduler]

trainer = pl.Trainer(
    gpus=1,                 # Number of GPUs to use
    precision=16,           # Use mixed precision (FP16)
    max_epochs=20,           # Number of epochs
    gradient_clip_val=0.5,  # Gradient clipping to prevent exploding gradients
    logger=WandbLogger(project="HiCur-NPC", name="Segment"),  # TensorBoard logger
    enable_checkpointing=False,
    callbacks=[            # List of callbacks
        ModelCheckpoint(
            dirpath="./model/seg/",
            filename="hicur_seg",
            monitor="mDice",
            mode="max",
            save_last=True
        ),
        RichProgressBar(leave = True),
        LearningRateMonitor(logging_interval='step')
    ]
)

model = Model()
trainer.fit(model)