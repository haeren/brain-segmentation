from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

mri_path = Path('NFBS_Dataset/A00028352/sub-A00028352_ses-NFB3_T1w.nii.gz')
label_path = Path('NFBS_Dataset/A00028352/sub-A00028352_ses-NFB3_T1w_brainmask.nii.gz')
brain_path = Path('NFBS_Dataset/A00028352/sub-A00028352_ses-NFB3_T1w_brain.nii.gz')

mri = nib.load(mri_path)
label = nib.load(label_path)
brain = nib.load(brain_path)

assert nib.aff2axcodes(mri.affine) == ('P', 'I', 'R')

mri_data = mri.get_fdata()
label_data = label.get_fdata().astype(np.uint8)
brain_data = brain.get_fdata()

plt.imshow(mri_data[:,:,95], cmap='gray') # cmap='bone'
plt.imsave('sample-slice-data.png', mri_data[:,:,95], cmap='gray') # cmap='bone'