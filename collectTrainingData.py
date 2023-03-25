import recordAudio
import rs_imaging

from pathlib import Path
import time
from datetime import datetime
import pyaudio

def checkImageFolders():
    Path("./Data/").mkdir(parents=True, exist_ok=True)
    Path("./Data/RS_Images/").mkdir(parents=True, exist_ok=True)
    Path("./Data/RS_IMages/jpg").mkdir(parents=True, exist_ok=True)
    Path("./Data/RS_Images/npy").mkdir(parents=True, exist_ok=True)
    Path("./Data/RS_Images/rgb-jpg").mkdir(parents=True, exist_ok=True)


'''
Parameter List
    Record Audio Inputs       
        rate:           sampling rate of recording
        chunk:          buffer size for recording use
        channels:       number of channels - should always be 1, we are recording in mono
        record_seconds: number of seconds to capture audio for
        saved_seconds:  number of seconds to save after syncing
        frame_shift:    number of frames to save before sync event

    Debug/Cycle Control:
        debug:          gives debug outputs if true
        cycles:         number of training samples to obtain
        delay_between:  delay between training cycles in seconds
        wait_for_input: waits for keyboard input if true between cycles

    Internal:
        file_id:        file_id of training sample: currently YYYYMMDD-HHMMSS
'''
if __name__ == '__main__':
    recordAudio.checkFolders()
    checkImageFolders()

    recordAudio.checkInputDevices()
    mic_ids = [-1,-1,-1]
    for i in range(3):
        mic_ids[i] = int(input("Enter Microphone {} id: ".format(i+1)))

    # Add microphone description for easy tracking    
    p = pyaudio.PyAudio()
    mic_desc = []
    for i in range(3):
        mic_desc.append(str("Mic " + str(i+1) + " ID: " +  str(mic_ids[i]) + " - " + p.get_device_info_by_host_api_device_index(0, i).get('name')))
    p.terminate()

    # Define parameters
    chirp_path = "./chirp3ms.wav"
    rate = 48000
    chunk = 4800
    channels = 1
    record_seconds = 1
    saved_seconds = 0.1 # 50 ms 
    frame_shift = 100

    cycles = 10
    debug = True
    delay_between = 0.1
    wait_for_input = False

    Imager = rs_imaging.RS_Imager()

    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    with open("./Data/Logs/" + curr_time + ".txt", 'w') as f:
        params = ["Chirp Path: " + str(chirp_path),
                    "Sampling Rate: " + str(rate),
                    "Chunk Size: " + str(chunk),
                    "Channels: " + str(channels),
                    "Record Seconds: " + str(record_seconds),
                    "Saved Seconds: " + str(saved_seconds),
                    "Frame Shift: " + str(frame_shift)]
        f.writelines(s + "\n" for s in mic_desc)
        f.writelines(s + "\n" for s in params)
        f.write('\n')
        f.write("List of file ids: \n")
        

        for i in range(cycles):
            if wait_for_input:
                input("Press Enter to Collect Next Sample")

            file_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            f.write(file_id + "\n")

            # Save images
            Imager.save_depth_image(file_id)
            Imager.save_RGB_image(file_id)

            # Save audio
            recordAudio.record(file_id, mic_ids, chirp_path, rate, chunk, channels,
                                record_seconds, saved_seconds, frame_shift, debug)

            if delay_between != 0:
                time.sleep(delay_between)