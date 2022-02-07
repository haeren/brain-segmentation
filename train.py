from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from dataset import BrainDataset
from model import UNet

augment_pipeline = iaa.Sequential([
        iaa.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),
        iaa.ElasticTransformation()
        ])

train_path = Path('Preprocessed/train/')
val_path = Path('Preprocessed/val/')

train_dataset = BrainDataset(train_path, augment_pipeline)
val_dataset = BrainDataset(val_path, None)

batch_size = 8
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_mask, actual_mask):
        pred = torch.flatten(pred_mask)
        actual = torch.flatten(actual_mask)
        
        numerator = (pred * actual).sum()
        denominator = pred.sum() + actual.sum() + 1e-8
        dice_score = (2 * numerator) / denominator
        return 1 - dice_score
        
class BrainSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_function = DiceLoss()
        
    def forward(self, data):
        return torch.sigmoid(self.model(data))
    
    def training_step(self, batch, batch_idx):
        slice_data, slice_mask = batch
        slice_mask = slice_mask.float()
        pred_mask = self(slice_data)
        loss = self.loss_function(pred_mask, slice_mask)
        self.log('Train Loss', loss)
        
        if batch_idx % 50 == 0:
            self.log_images(slice_data.cpu(), pred_mask.cpu(), slice_mask.cpu(), 'Train')
            
        return loss

    def validation_step(self, batch, batch_idx):
        slice_data, slice_mask = batch
        slice_mask = slice_mask.float()
        pred_mask = self(slice_data)
        loss = self.loss_function(pred_mask, slice_mask)
        self.log('Val Loss', loss)
        
        if batch_idx % 2 == 0:
            self.log_images(slice_data.cpu(), pred_mask.cpu(), slice_mask.cpu(), 'Val')
            
        return loss
        
    def log_images(self, slice_data, pred_mask, slice_mask, name):
        pred_mask = pred_mask > 0.5
        fig, axis = plt.subplots(1, 2)
        
        axis[0].imshow(slice_data[0][0], cmap='bone')
        actual_mask_ma = np.ma.masked_where(slice_mask[0][0] == 0, slice_mask[0][0])
        axis[0].imshow(actual_mask_ma, alpha=0.6)
        
        axis[1].imshow(slice_data[0][0], cmap='bone')
        pred_mask_ma = np.ma.masked_where(pred_mask[0][0] == 0, pred_mask[0][0])
        axis[1].imshow(pred_mask_ma, alpha=0.6)
        
        self.logger.experiment.add_figure(name, fig, self.global_step)
        
    def configure_optimizers(self):
        return [self.optimizer]

# Initialize same random values as weights
torch.manual_seed(0)
model = BrainSegmentation()

checkpoint_callback = ModelCheckpoint(monitor='Val Loss', save_top_k=10, mode='min')

# Assign 0 for CPU training
gpus = 1
trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir='./logs'),
                     log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=50)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)