from pathlib import Path
from train import BrainSegmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from preprocess import standardize, normalize

model_path = Path('Logs/default/version_0/checkpoints/epoch=20-step=37799.ckpt')
model = BrainSegmentation.load_from_checkpoint(model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

sample_path = Path('NFBS_Dataset/A00060516/sub-A00060516_ses-NFB3_T1w.nii.gz')
mri = nib.load(sample_path)
assert nib.aff2axcodes(mri.affine) == ('P', 'I', 'R')
mri_data = mri.get_fdata()
mri_data = standardize(mri_data)
mri_data = normalize(mri_data)

pred_masks = []
model.eval()

for i in range(mri_data.shape[-1]):
    slice = mri_data[:,:,i] # Get i th slice
    with torch.no_grad():
        slice_data = torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device) # Add Batch and Channel dims
        pred_mask = model(slice_data)[0,0]  # Ignore Batch and Channel dims
        pred_mask = pred_mask > 0.5         # Apply threshold before saving the mask as image
    pred_masks.append(pred_mask.cpu())

save_root = Path('TestResults')        
for i in range(mri_data.shape[-1]):
    brain = np.ma.masked_where(pred_masks[i]==0, mri_data[:,:,i])
    brain_path = save_root/'Brain'
    brain_path.mkdir(parents=True, exist_ok=True)
    plt.imsave(brain_path/(str(i) + '.png'), brain, cmap='gray')