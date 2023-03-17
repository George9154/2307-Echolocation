import pyrealsense2 as rs
import cv2
import numpy as np
import time

'''
This file provides functions for using the Realsense camera.

Full documentation for pyrealsense2 (its kinda bad tho):
    https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#module-pyrealsense2

'''

class RS_Imager:

    def __init__(self):
        # Instantiate an RS_Imager object

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)


    def save_depth_image(self, save_array=True, hole_filling_mode=1):
        '''
            Purpose: Saves a depth image using the RealSense camera. Can optionally also save the image as an ndarray.
            Note: Assumes that these folders already exist:
                ./Data/RS_Images/jpg
                ./Data/RS_Images/npy

            Arg - filepath (string): the complete filepath and name of the image to be saved
            Arg - hole_filling_mode (int): determines if and how holes are filled. Options:
                        -1  No hole filling
                        0   Fill holes from the left
                        1   Fill holes with the deepest pixel that is nearby.
                            
            Returns: an ndarray corresponding to the depth image
        '''

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Apply hole filling if needed
        if hole_filling_mode >= 0:
            hole_filler = rs.hole_filling_filter(hole_filling_mode)
            depth_frame = hole_filler.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)

        curr_time = time.strftime("%Y%m%d-%H%M%S")

        cv2.imwrite("./Data/RS_Images/jpg/" + curr_time + ".jpg", depth_colormap)
        
        if save_array:
            np.save("./Data/RS_Images/npy/" + curr_time + ".npy", depth_image)

        return depth_image