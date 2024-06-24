import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
import torch
import torch.optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import cv2
from EandD import HCMAE  # Import your HCMAE model class
from einops import rearrange

# Configurations
config = {
    'batch_size': 128,
    'image_size': 384,
    'train_fold': 0, # fold-0 for training and fold-1 for validation
    'crop_size': (384, 384),
    'base_learning_rate': 1e-4,
    'weight_decay': 5e-2,
    'warmup_epoch': 100,
    'total_epoch': 1000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mask_ratio': 0.75,
    'df_path': './Data/HCMAENPC/df.csv',
    'model_path': '/pth/hcmae.pth',
    'data_dir': '/Data/HCMAENPC/imgs/',
    'mean': [0.53017267, 0.31335013, 0.31556818],
    'std': [0.33659712, 0.23721003, 0.24286168],
}

# Transforms
train_trans = A.Compose([
    A.Resize(config['image_size'], config['image_size']),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.ColorJitter(),
    A.Normalize(mean=config['mean'], std=config['std']),
    ToTensorV2()
])

valid_trans = A.Compose([
    A.Resize(config['image_size'], config['image_size']),
    A.Normalize(mean=config['mean'], std=config['std']),
    ToTensorV2()
])


class NasoDataset(Dataset):
    def __init__(self, df, phase, trans=None):
        self.df = df
        self.phase = phase
        self.trans = trans

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_file = config['data_dir'] + row["image"]
        img1 = cv2.imread(image_file)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        label = int(row["label"])

        if self.trans is not None:
            aug1 = self.trans(image=img1)
            img1 = aug1['image']

        return img1, label

    def __len__(self):
        return self.df.shape[0]


def create_datasets():
    df = pd.read_csv(config['df_path']).iloc[0:1000]
    train_idx = np.where(df.fold == config['train_fold'])[0]
    valid_idx = np.where(df.fold != config['train_fold'])[0]

    ds_train = NasoDataset(df.loc[train_idx].reset_index(drop=True), "train", train_trans)
    ds_valid = NasoDataset(df.loc[valid_idx].reset_index(drop=True), "val", valid_trans)

    return ds_train, ds_valid


def train_model(model, dataloader, total_epoch, model_path):
    optim = torch.optim.AdamW(model.parameters(), lr=config['base_learning_rate'] * config['batch_size'] / 256, betas=(0.9, 0.95), weight_decay=config['weight_decay'])
    lr_func = lambda epoch: min((epoch + 1) / (config['warmup_epoch'] + 1e-8), 0.5 * (np.cos(epoch / config['total_epoch'] * np.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    wandb.init(project='HiCur-NPC', name='HCMAE', config=config)

    for e in range(total_epoch):
        model.train()
        losses = []
        train_step = len(dataloader)

        with tqdm(total=train_step, desc=f'Epoch {e + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for step, (img, label) in enumerate(dataloader):
                img = img.to(config['device'])
                loss = model.forward_loss(img)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())

                pbar.set_postfix(**{'Train Loss': np.mean(losses)})
                pbar.update(1)

        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        wandb.log({'mae_loss': avg_loss}, step=e)
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        # Visualize the first 16 predicted images on the validation dataset
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([ds_valid[i][0] for i in range(16)])
            val_img = val_img.to(config['device'])
            _, predicted_val_img, mask = model.forward_vit(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            wandb.log({'hcmae_image': wandb.Image((img + 1) / 2)}, step=e)

        # Save model
        torch.save(model.module if hasattr(model, 'module') else model, model_path)  # Save the non-parallelized model if applicable

    wandb.finish()


# Main script execution
if __name__ == "__main__":
    ds_train, ds_valid = create_datasets()
    dataloader = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    model = HCMAE().to(config['device'])
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    train_model(model, dataloader, config['total_epoch'], config['model_path'])
