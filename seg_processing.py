import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image

def load_clean_seg(path):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(path))
    seg = seg.astype(np.int)
    seg[seg < 2884] = 0
    unique = np.unique(seg)
    for i,e in enumerate(unique):
        seg[seg==e] = i
    return seg

def hot_encode(arr):
    
    sx, sy, sz = arr.shape
    sc = arr.max()
    output = np.zeros((sc, sx, sy, sz))
    for i in range(1, sc+1):
        output[i-1][arr==i] = 1

    print(output.shape)

if __name__=="__main__":

    arr = load_clean_seg("/vol/biomedic3/hjr119/XCAT/generation/samp3_SEG_1.nii.gz")
    hot_encode(arr)
    exit()

    ct_itk = sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/samp0_CT_1.nii.gz")
    ct  = sitk.GetArrayFromImage(ct_itk)
    seg = sitk.GetArrayFromImage(sitk.ReadImage("/vol/biomedic3/hjr119/XCAT/generation/samp0_SEG_1.nii.gz"))
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


    # # seg[seg < 2884] = 0
    # # seg[seg > 2910] = 0
    # seg[seg != 2885] = 0

    seg[seg < 2884] = 0


    sitk_arr = sitk.GetImageFromArray(seg)
    sitk_arr.SetSpacing([pix_res, pix_res, pix_res])
    writer = sitk.ImageFileWriter()
    writer.SetFileName("/vol/biomedic3/hjr119/XCAT/generation/test_seg.nii.gz")
    writer.Execute(sitk_arr)

    # ct[seg == 0] = 0
    # ct = (ct-ct.min()) / (ct.max()-ct.min())

    # sitk_arr = sitk.GetImageFromArray(ct)
    # sitk_arr.SetSpacing([pix_res, pix_res, pix_res])
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName("/vol/biomedic3/hjr119/XCAT/generation/heart_ct_focus.nii.gz")
    # writer.Execute(sitk_arr)


    print("done")