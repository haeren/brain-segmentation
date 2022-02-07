from pathlib import Path
from train import BrainSegmentation, DiceLoss
import torch
from dataset import BrainDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = Path('Logs/default/version_0/checkpoints/epoch=20-step=37799.ckpt')

model = BrainSegmentation.load_from_checkpoint(model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_path = Path('Preprocessed/test/')
test_dataset = BrainDataset(test_path, None)

pred_masks = []
slice_masks = []

model.eval()
 
for slice_data, slice_mask in tqdm(test_dataset):
    slice_data = torch.tensor(slice_data).to(device).unsqueeze(0) # Add Batch dim to input with unsqueeze since there is no loader
    with torch.no_grad():
        pred_mask = model(slice_data)
    pred_masks.append(pred_mask.cpu().numpy())
    slice_masks.append(slice_mask)
    
pred_masks = np.array(pred_masks)
slice_masks = np.array(slice_masks)

dice_score = 1 - DiceLoss()(torch.from_numpy(pred_masks), torch.from_numpy(slice_masks).unsqueeze(0).float())
print(f'Dice Score:', {dice_score})

save_root = Path('TestResults')

for i in range(len(pred_masks)):
    pred_mask = pred_masks[i,0,0]   # Ignore Batch and Channel dims
    pred_mask = pred_mask > 0.5     # Apply threshold before saving the mask as image
    slice_mask = slice_masks[i,0]   # Ignore Channel dim
    pred_mask_path = save_root/'PredMask'
    slice_mask_path = save_root/'SliceMask'
    pred_mask_path.mkdir(parents=True, exist_ok=True)
    slice_mask_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(pred_mask_path/(str(i) + '.png'), pred_mask, cmap='gray')
    plt.imsave(slice_mask_path/(str(i) + '.png'), slice_mask, cmap='gray')