import os
import nibabel as nib
import numpy as np
from pathlib import Path

def standardize(volume):
    mean = volume.mean()
    std = np.std(volume)
    standardized = (volume - mean) / std
    return standardized

def normalize(standardized):
    normalized = (standardized - standardized.min()) / (standardized.max() - standardized.min())
    return normalized

path = Path('NFBS_Dataset')
save_root = Path('Preprocessed')

if __name__ == '__main__':
    for root, dirs, files in os.walk(path):
        for file in files:
            file_type = None
            if file.endswith('T1w.nii.gz'):
                file_type = 'Data'
            elif file.endswith('T1w_brainmask.nii.gz'):
                file_type = 'Mask'
            else:
                continue
            
            file_path = os.path.join(root, file)
            mri = nib.load(file_path)
            assert nib.aff2axcodes(mri.affine) == ('P', 'I', 'R')
            
            if file_type == 'Data':
                mri_data = mri.get_fdata()
                mri_data = standardize(mri_data)
                mri_data = normalize(mri_data)
                
            elif file_type == 'Mask':
                mri_data = mri.get_fdata().astype(np.uint8)
                
            sample_name = Path(root).parts[-1]
            
            for i in range(mri_data.shape[-1]):
                slice = mri_data[:,:,i]
                save_path = save_root/sample_name/file_type
                save_path.mkdir(parents=True, exist_ok=True)
                np.save(save_path/str(i), slice)