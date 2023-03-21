#!/usr/bin/env python

'''
This is an example of how to use rs_imaging.py

'''

import rs_imaging
from pathlib import Path
import time

Path("./Data/").mkdir(parents=True, exist_ok=True)
Path("./Data/RS_Images/").mkdir(parents=True, exist_ok=True)
Path("./Data/RS_IMages/jpg").mkdir(parents=True, exist_ok=True)
Path("./Data/RS_Images/npy").mkdir(parents=True, exist_ok=True)
Path("./Data/RS_Images/rgb-jpg").mkdir(parents=True, exist_ok=True)


Imager = rs_imaging.RS_Imager()

curr_time = time.strftime("%Y%m%d-%H%M%S")

Imager.save_depth_image(curr_time, save_array=True)
Imager.save_RGB_image(curr_time)
