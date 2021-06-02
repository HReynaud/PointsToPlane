import os
import numpy as np
import sys
import time
import subprocess
import SimpleITK as sitk
from PIL import Image
from subprocess import Popen

commands = ['python gen_xcat.py samp'+str(i) for i in range(10)]
print(commands)

procs = [ Popen(i.split(" ")) for i in commands ]
for p in procs:
   p.wait()

print("done")
