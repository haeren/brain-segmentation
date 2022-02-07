from pathlib import Path
import torch
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, augment_params):
        self.slice_data_paths = self.get_data_paths(path)
        self.augment_params = augment_params
        
    @staticmethod
    def get_data_paths(path):
        slice_data_paths = []
        for sample in path.glob('*'):
            data_path = sample/'Data'
            for slice_data_path in data_path.glob('*.npy'):
                slice_data_paths.append(slice_data_path)
        return slice_data_paths
            
    @staticmethod
    def get_mask_path(slice_data_path):
        parts = list(slice_data_path.parts)
        parts[parts.index('Data')] = 'Mask'
        return Path(*parts)
    
    def augment(self, slice_data, slice_mask):
        random_seed = torch.randint(0, 1000000, (1,)).item()
        imgaug.seed(random_seed)
        slice_mask = SegmentationMapsOnImage(slice_mask, slice_mask.shape)
        slice_data_aug, slice_mask_aug = self.augment_params(image=slice_data, segmentation_maps=slice_mask)
        slice_mask_aug = slice_mask_aug.get_arr()
        return slice_data_aug, slice_mask_aug
    
    def __len__(self):
        return len(self.slice_data_paths)
    
    def __getitem__(self, idx):
        slice_data_path = self.slice_data_paths[idx]
        slice_mask_path = self.get_mask_path(slice_data_path)
        slice_data = np.load(slice_data_path).astype(np.float32)
        slice_mask = np.load(slice_mask_path)
        
        if self.augment_params:
            slice_data, slice_mask = self.augment(slice_data, slice_mask)
            
        # Pytorch input shape: BxCxHxW
        # B: Batch, C: Channel, H: Height, W: Width
        # Add C dim with np.expand_dims
        # B dim will be added by data loader
        return np.expand_dims(slice_data, 0), np.expand_dims(slice_mask, 0)

"""
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

augment_pipeline = iaa.Sequential([
        iaa.Affine(scale=(0.85, 1.15), rotate=(-45, 45)),
        iaa.ElasticTransformation()
        ])

path = Path('Preprocessed/')
dataset = BrainDataset(path, augment_pipeline)

fig, axis = plt.subplots(3, 3, figsize=(9, 9))

for i in range(3):
    for j in range(3):
        slice_data, slice_mask = dataset[42]
        slice_mask_ma = np.ma.masked_where(slice_mask==0, slice_mask)
        axis[i][j].imshow(slice_data[0], cmap='bone')
        axis[i][j].imshow(slice_mask_ma[0], cmap='autumn')
        axis[i][j].axis('off')
        
fig.suptitle('Sample augmentations')
plt.tight_layout()
"""