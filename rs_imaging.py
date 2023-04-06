import pyrealsense2 as rs
import cv2
import numpy as np

'''
This file provides functions for using the Realsense camera.

Full documentation for pyrealsense2 (its kinda bad tho):
    https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#module-pyrealsense2

'''
RESOLUTION = (128, 128)

class RS_Imager:

    def __init__(self):
        # Instantiate an RS_Imager object

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)


    def save_depth_image(self, file_ID, save_array=True, debug_mode=False, hole_filling_mode=-1):
        '''
            Purpose: Saves a depth image using the RealSense camera. Can optionally also save the image as an ndarray.
            Note: Assumes that these folders already exist:
                ./Data/RS_Images/jpg
                ./Data/RS_Images/npy

            Arg - file_ID (string): this is the name of the file(s) to be saved. Do NOT include the file extension.
            Arg - save_array (boolean): determines if an ndarray should be saved as a .npy file as well as the .jpg
            Arg - debug_mode (boolean): determines if full resolution images should be saved in addition to low-res.
            Arg - hole_filling_mode (int): determines if and how holes are filled. Options:
                        -1  No hole filling
                        0   Fill holes from the left
                        1   Fill holes with the deepest pixel that is nearby.
                            
            Returns: an ndarray corresponding to the depth image, in the reduced resolution.
        '''

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Apply hole filling if needed
        if hole_filling_mode >= 0:
            hole_filler = rs.hole_filling_filter(hole_filling_mode)
            depth_frame = hole_filler.process(depth_frame)

        full_res_array = np.asanyarray(depth_frame.get_data())
        full_res_image = cv2.applyColorMap(cv2.convertScaleAbs(full_res_array, alpha=0.03), cv2.COLORMAP_RAINBOW)

        low_res_image = cv2.resize(full_res_image, RESOLUTION)

        if file_ID != None:
            cv2.imwrite("./Data/RS_Images/jpg/" + file_ID + ".jpg", low_res_image)

            if debug_mode:
                cv2.imwrite("./Data/RS_Images/jpg/" + file_ID + "-full_res.jpg", full_res_image)
            
            if save_array:
                low_res_array = cv2.resize(full_res_array, RESOLUTION)
                np.save("./Data/RS_Images/npy/" + file_ID + ".npy", low_res_array)
                
                if debug_mode:
                    np.save("./Data/RS_Images/npy/" + file_ID + "-full_res.npy", full_res_array)

        return full_res_array


    def save_RGB_image(self, file_ID):
        '''
            Purpose: Saves an RGB image using the RealSense camera. Can optionally also save the image as an ndarray.
            Note: Assumes that these folders already exist:
                ./Data/RS_Images/jpg-rgb
            
            Arg - file_ID (string): this is the name of the file(s) to be saved. Do NOT include the file extension.  
            
            Returns: an ndarray corresponding to the rgb image
        '''

        frames = self.pipeline.wait_for_frames()
        rgb_frame = frames.get_color_frame()
        image = np.asanyarray(rgb_frame.get_data())

        if file_ID != None:
            cv2.imwrite("./Data/RS_Images/rgb-jpg/" + file_ID + ".jpg", image)

        return image
