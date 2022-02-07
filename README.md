# Brain Segmentation
PyTorch implementation of U-Net for brain segmentation from MRI slices

## Dataset
The Neurofeedback Skull-stripped (NFBS) repository:
http://preprocessed-connectomes-project.org/NFB_skullstripped/

Directory and file structure should look like this before train/test (train/val/test are three separate folders under Preprocessed):
```
Preprocessed
|---(train/val/test)
    |---sample1
    |   |---Data
    |   |       slice1.npy
    |   |       sliceN.npy
    |   |---Mask
    |           slice1.npy
    |           sliceN.npy
    |---sampleN
        |---Data
        |       slice1.npy
        |       sliceN.npy
        |---Mask
                slice1.npy
                sliceN.npy
```

## References
- O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," in International Conference on Medical image computing and computer-assisted intervention, 2015: Springer, pp. 234-241.
- B. Puccio, J. P. Pooley, J. S. Pellman, E. C. Taverna, and R. C. Craddock, "The preprocessed connectomes project repository of manually corrected skull-stripped T1-weighted anatomical MRI data," Gigascience, vol. 5, no. 1, pp. s13742-016-0150-5, 2016.
