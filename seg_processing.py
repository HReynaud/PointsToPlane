import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image

ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/default_512_CT_1.nii.gz")
ct  = sitk.GetArrayFromImage(ct_itk)
seg = sitk.GetArrayFromImage(sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/default_512_SEG_1.nii.gz"))
pix_res = ct_itk.GetSpacing()[0]

print("CT  shape", ct.shape)
print("Seg shape", seg.shape)

unique, counts = np.unique(seg, return_counts=True)

# Fill dict
names = dict()
names[0] = "void"
with open("/vol/biomedic3/hjr119/XCAT/program/organ_ids.txt") as f:
    content = f.readlines()
    for l in content[1:]:
        l = l.replace('\n', '')
        elements = l.split(' = ')
        names[int(elements[1])] = elements[0]


for u, c in zip(unique, counts):
    print(names[int(u)], int(u), c)


# seg[seg < 2884] = 0
# seg[seg > 2910] = 0
seg[seg != 2885] = 0

sitk_arr = sitk.GetImageFromArray(seg)
sitk_arr.SetSpacing([pix_res, pix_res, pix_res])
writer = sitk.ImageFileWriter()
writer.SetFileName("/vol/biomedic3/hjr119/XCAT/generation/default_512_SEG_1_FOCUS.nii.gz")
writer.Execute(sitk_arr)

ct[seg == 0] = 0
ct = (ct-ct.min()) / (ct.max()-ct.min())

sitk_arr = sitk.GetImageFromArray(ct)
sitk_arr.SetSpacing([pix_res, pix_res, pix_res])
writer = sitk.ImageFileWriter()
writer.SetFileName("/vol/biomedic3/hjr119/XCAT/generation/heart_ct_focus.nii.gz")
writer.Execute(sitk_arr)


print("done")