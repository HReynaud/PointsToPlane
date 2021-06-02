import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image
from numpy.random import default_rng

############ START OF PARAMS ############
# Generation time:
# 256 = 47s 
# 512 = 10 minutes
# 1024 = 30 minutes

final_dim = 512
CT_OR_SEG = 'CT'
tname = "SMOL" #sys.argv[1]
name = tname+"_"+CT_OR_SEG

generate = True
clean_bin = True

slice_precentage_start = 1.-.18 # 0
slice_precentage_stop  = 1.-.08 # 1 - 0.11
# WARNING: These depends on the ZOOM factor
# which is defined below
ZOOM = 4
dim_2_percentage_start = 0.
dim_2_percentage_stop  = .75
dim_3_percentage_start = .2
dim_3_percentage_stop  = .95

# Heart adjustements
use_random_sensible = True         # If set to true, all values will be randomized in while staying in a sensible range

apical_thin = 0.0	                # apical_thinning (0 to 1.0 scale, 0.0 = not present, 0.5 = halfway present, 1.0 = completely thin)
uniform_heart = 0	                # sets the thickness of the LV (0 = default, nonuniform wall thickness; 1 = uniform wall thickness for LV)
hrt_start_ph_index = 0.0	        # hrt_start_phase_index (range=0 to 1; ED=0, ES=0.4) see NOTE 3 
heart_base = "vmale50_heart.nrb"	# basename for heart files (male = vmale50_heart.nrb; female = vfemale50_heart.nrb)
lv_radius_scale = 1		            # lv_radius_scale (value from 0 to 1 to scale the radius of the left ventricle)
lv_length_scale = 1		            # lv_length_scale (value from 0 to 1 to scale the length of the left ventricle)
hrt_scale_x = 1.0		            # hrt_scale x  
hrt_scale_y = 1.0		            # hrt_scale y  
hrt_scale_z = 1.0		            # hrt_scale z  
motion_defect_flag = 0	            # (0 = do not include, 1 = include) regional motion abnormality in the LV as defined by heart lesion parameters see NOTE 9


#Generate a sequence
out_period = 1		                # output_period (SECS) (if <= 0, then output_period=time_per_frame*output_frames)
out_frames = 1		                # output_frames (# of output time frames )

############## END OF PARAMS ##############



################################################
##### DO NOT MODIFY PARAMS BELOW THIS LINE #####
################################################
if CT_OR_SEG == 'CT':
    get_act = 0
    get_seg = 0
    get_atn = 1
else:
    get_act = 1
    get_seg = 1
    get_atn = 0
    
plane_dim = int(np.ceil(final_dim / (dim_2_percentage_stop-dim_2_percentage_start)))

path = "/vol/biomedic3/hjr119/XCAT/program"
os.chdir(path)
BASE = 256/ZOOM # Default
CM_TO_PIX = 0.3125*BASE 
SLICE_MIN = 1
SLICE_MAX = 500*(plane_dim/BASE)
pix_res = CM_TO_PIX/plane_dim
slice_bot = int(max(SLICE_MIN, np.floor(SLICE_MAX*slice_precentage_start)))
slice_top = int(min(SLICE_MAX, np.floor(SLICE_MAX*slice_precentage_stop)))

if use_random_sensible:
    rng = default_rng() # No fixed seed for multi threading

    apical_thin = np.round(rng.random()*1.1, 1) 
    uniform_heart = 1 if rng.random() > 0.9 else 0 # 10% chance	                
    hrt_start_ph_index = np.round(rng.random()*1.1, 1)
    heart_base = rng.choice(["vmale50_heart.nrb", "vfemale50_heart.nrb"])
    lv_radius_scale = min(1.0, rng.lognormal(0.05, 0.3, None))
    lv_length_scale = min(1.0, rng.lognormal(0.05, 0.3, None))
    hrt_scale_x = min(1.0, rng.lognormal(0.05, 0.3, None)) 
    hrt_scale_y = min(1.0, rng.lognormal(0.05, 0.3, None))
    hrt_scale_z = min(1.0, rng.lognormal(0.05, 0.3, None))
    motion_defect_flag = 1 if rng.random() > 0.9 else 0 # 10% chance	 



# Phantom generation
start = time.time()
bashCmd =   "./dxcat2_linux_64bit general.samp.par "            + \
            "--startslice "         + str(slice_bot)            +" "+ \
            "--endslice "           + str(slice_top)            +" "+ \
            "--pixel_width "        + str(pix_res)              +" "+ \
            "--slice_width "        + str(pix_res)              +" "+ \
            "--array_size "         + str(plane_dim)            +" "+ \
            "--act_phan_each "      + str(get_act)              +" "+ \
            "--out_frames "         + str(out_frames)           +" "+ \
            "--color_code "         + str(get_seg)              +" "+ \
            "--atten_phan_each "    + str(get_atn)              +" "+ \
            "--apical_thin "        + str(apical_thin)          +" "+ \
            "--uniform_heart "      + str(uniform_heart)        +" "+ \
            "--hrt_start_ph_index " + str(hrt_start_ph_index)   +" "+ \
            "--heart_base "         + str(heart_base)           +" "+ \
            "--lv_radius_scale "    + str(lv_radius_scale)      +" "+ \
            "--lv_length_scale "    + str(lv_length_scale)      +" "+ \
            "--hrt_scale_x "        + str(hrt_scale_x)          +" "+ \
            "--hrt_scale_y "        + str(hrt_scale_y)          +" "+ \
            "--hrt_scale_z "        + str(hrt_scale_z)          +" "+ \
            "--motion_defect_flag " + str(motion_defect_flag)   +" "+ \
            "../generation/"+name

print(bashCmd)
output = '\\n'*10
if generate:
    process = subprocess.Popen(bashCmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
stop = time.time()
print("Generated in ", stop-start, "seconds.")
# print("Generated slice ",slice," in ", stop-start, "s")
print(str(output).split('\\n')[-3])

canvas = None

for i in range(1, out_frames+1):
    # Openning binary file
    print("Loading binary",i,"...")
    ext = "_atn_" if CT_OR_SEG == 'CT' else "_act_"
    f=open("/vol/biomedic3/hjr119/XCAT/generation/"+name+ext+str(i)+".bin", "rb")
    content = f.read()

    # Transforming binary to numpy array
    print("Transforming binary to numpy...")
    arr = np.frombuffer(content, dtype=np.float32)
    arr = arr.reshape(slice_top-slice_bot+1, plane_dim, plane_dim)
    arr = np.flip(arr, axis=0) # put feets at the bottom

    # Delete content to save ram
    content = None

    # Adjusting array dimensions to focus on the heart
    sx, sy, sz = arr.shape
    d2s, d2e, d3s, d3e = int(dim_2_percentage_start*sy), int(dim_2_percentage_stop*sy), int(dim_3_percentage_start*sz), int(dim_3_percentage_stop*sz)
    arr = arr[:(d2e-d2s), d2s:d2e, d3s:d3e]
    print()
    print("Final array size is", arr.shape, "with pix res:", pix_res)
    print()

    # Transforming numpy to sitk
    print("Transforming numpy to sitk...")
    sitk_arr = sitk.GetImageFromArray(arr)
    sitk_arr.SetSpacing([pix_res, pix_res, pix_res])

    # Save sitk to nii.gz file
    print("Saving sitk to .nii.gz file...", )
    writer = sitk.ImageFileWriter()
    writer.SetFileName("/vol/biomedic3/hjr119/XCAT/generation/"+name+"_"+str(i)+".nii.gz")
    writer.Execute(sitk_arr)
    if clean_bin:
        if os.path.exists("/vol/biomedic3/hjr119/XCAT/generation/"+name+ext+str(i)+".bin"):
            os.remove("/vol/biomedic3/hjr119/XCAT/generation/"+name+ext+str(i)+".bin")

    # Generate sample
    sx, sy, sz = arr.shape
    middle_cut = arr[:,sy//2,:]
    middle_cut = middle_cut/middle_cut.max()*255

    if type(canvas) == type(None):
        canvas = middle_cut
    else:
        canvas = np.concatenate((canvas, middle_cut), axis=1)

Image.fromarray(canvas.astype(np.uint8)).save("/vol/biomedic3/hjr119/XCAT/generation/"+name+"_sample.jpg", quality=100)






