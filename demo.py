'''
Code to demonstrate producing an image from sound

Created 2023-04-06 by Jamen (but copying lots of George's code)

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
    record_seconds = 1
    saved_seconds = 0.05 # 50 ms
    frame_shift = 100

    cycles = 1
    debug = False
    sync_signal = True
    delay_between = 0.2
    wait_for_input = False
    capture_RGB_image = False
    capture_depth_image = False
    
    # Get the audio signals
    
    
    audioArray = recordAudio.record(None, mic_ids, chirp_path, rate, chunk, channels,
                                record_seconds, saved_seconds, frame_shift, debug, sync_signal, saveData = False)

    print("Captured audio signals")
    
    # ML stuff
    model = WaveformNet("direct")
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        x = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[0, :],axis=0),axis=0), axis=0)
        y = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[1, :],axis=0),axis=0), axis=0)
        z = np.expand_dims(np.expand_dims(np.expand_dims(audioArray[2, :],axis=0),axis=0), axis=0)
        # x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        # z = torch.from_numpy(z)
        print(x.shape)
        w = model.forward(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)).detach().numpy()
    
    print(type(w))
        
    plt.imshow(w[0, 0, :, :], "gray")
    plt.show()

    cv2.imshow("Predicted Image", w[0, 0, :, :])
    
    # If applicable, capture and displey depth and RGB images
    if capture_RGB_image or capture_depth_image:
        Imager = rs_imaging.RS_Imager()
        
        if capture_depth_image:
            depth_array = Imager.save_depth_image(file_ID=None)
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.03), cv2.COLORMAP_RAINBOW)
            cv2.imshow("Depth Image", depth_image)
            
        if capture_RGB_image:
            rgb_image = Imager.save_RGB_image(file_ID=None)
            cv2.imshow("RGB Image", rgb_image)
            
    cv2.waitKey()
