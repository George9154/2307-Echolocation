'''
Code to demonstrate producing an image from sound
Note: Press q to shutdown.

Created 2023-04-06

Outline: 
    0. Set params
        - Setting for if it should take a depth/rgb image for comparison (want to be able to run with or without camera)
    1. Emit the chirp and collect waves with recordAudio.py
    3. Put sound through NN
    4. Display the predicted image, as well as depth/rgb image if applicable

    ^ Do this first, then look at making it loop to take video

'''

import recordAudio
import pyaudio
import rs_imaging
import cv2
import numpy as np
import torch
from model import WaveformNet
import matplotlib.pyplot as plt
import time

print("Done imports")

if __name__ == '__main__':

    recordAudio.checkInputDevices()
    mic_ids = [-1,-1,-1]
    for i in range(3):
        mic_ids[i] = int(input("Enter Microphone {} id: ".format(i+1)))
        
    # Add microphone description for easy tracking    
    p = pyaudio.PyAudio()
    mic_desc = []
    for i in range(3):
        mic_desc.append(str("Mic " + str(i+1) + " ID: " +  str(mic_ids[i]) + " - " + p.get_device_info_by_host_api_device_index(0, mic_ids[i]).get('name')))
    p.terminate()
    

    # Define parameters
    model_path = "NN_models\model_1.pt"
    
    chirp_path = "./Chirps/downchirp5ms.wav"
    rate = 48000
    chunk = 4800
    channels = 1
    record_seconds = 0.5
    saved_seconds = 0.05 # 50 ms
    frame_shift = 100

    cycles = 1
    debug = False
    sync_signal = True
    delay_between = 0.2
    wait_for_input = False
    show_RGB_frames = False
    show_depth_frames = False
    resolution = (512, 512)
    
    # fig, ax = plt.subplots()
    # plt.show(block=False)
    # time.sleep(10)
    
    while(True):
        # Get the audio signals
        audioArray = recordAudio.record(None, mic_ids, chirp_path, rate, chunk, channels,
                                    record_seconds, saved_seconds, frame_shift, debug, sync_signal, saveData = False)
        
        # ML stuff
        model = WaveformNet("direct")
        model.load_state_dict(torch.load(model_path))

        model.eval()

        with torch.no_grad():
            x = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[0, :],axis=0),axis=0), axis=0)
            y = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[1, :],axis=0),axis=0), axis=0)
            z = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[2, :],axis=0),axis=0), axis=0)
            w = model.forward(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)).detach().numpy()
            
        # ax.imshow(w[0, 0, :, :], "gray")
        # plt.show(block=False)
        # time.sleep(3)
        predicted_frame = cv2.resize(w[0, 0, :, :], resolution)

        cv2.imshow("Predicted Image - press \'q\' to shutdown", predicted_frame)
        
        # If applicable, capture and displey depth and RGB images
        if show_RGB_frames or show_depth_frames:
            Imager = rs_imaging.RS_Imager()
            
            if show_depth_frames:
                depth_array = Imager.save_depth_image(file_ID=None)
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.03), cv2.COLORMAP_RAINBOW)
                cv2.imshow("Depth Image", depth_image)
                
            if show_RGB_frames:
                rgb_image = Imager.save_RGB_image(file_ID=None)
                cv2.imshow("RGB Image", rgb_image)
    
        if cv2.waitKey(1) == ord('q'):
            break
                
    cv2.destroyAllWindows()
    