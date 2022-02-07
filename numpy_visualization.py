from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

slice_data_path = Path('Preprocessed/A00028185/Data')
slice_mask_path = Path('Preprocessed/A00028185/Mask')

slice_no = 95

slice_data = np.load(slice_data_path/(str(slice_no)+'.npy'))
slice_mask = np.load(slice_mask_path/(str(slice_no)+'.npy'))

plt.figure()
plt.imshow(slice_data, cmap='bone')
slice_mask_ma = np.ma.masked_where(slice_mask==0, slice_mask)
plt.imshow(slice_mask_ma, cmap='autumn')

print('Slice data min:', slice_data.min())
print('Slice data max:', slice_data.max())
print('Slice mask min:', slice_mask.min())
print('Slice mask max:', slice_mask.max())
print('Slice mask MA min:', slice_mask_ma.min())
print('Slice mask MA max:', slice_mask_ma.max())

is_max = False

for i in range(192):
    slice_data = np.load(slice_data_path/(str(i)+'.npy'))
    if slice_data.max() == 1:
        is_max = True
        
print(is_max)