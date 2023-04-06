import time
import wave
import pyaudio
import numpy as np
from matplotlib import pyplot as plt

from scipy.io.wavfile import write
from pathlib import Path


# Lists all input audio devices
def checkInputDevices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


# Checks if folders exist and creates them if not 
def checkFolders():
    Path("./Data/").mkdir(parents=True, exist_ok=True)
    Path("./Data/wav/").mkdir(parents=True, exist_ok=True)
    Path("./Data/npy/").mkdir(parents=True, exist_ok=True)
    Path("./Data/Plots/").mkdir(parents=True, exist_ok=True)
    Path("./Data/Logs/").mkdir(parents=True, exist_ok=True)


# Sends out a chirp and records 
# Inputs:   mics: List of 3 microphone ids
#           savewav: Saves the recorded audio as wav files
#           plotting: Saves plots of the recorded audio 
def record(file_id, mic_ids, chirp_path, rate, chunk, channels,
            record_seconds, saved_seconds, frame_shift, debug, sync_signal = True, saveData = True):
    
    format = pyaudio.paFloat32

    mic1_id = int(mic_ids[0])
    mic2_id = int(mic_ids[1])
    mic3_id = int(mic_ids[2])

    with wave.open(chirp_path, 'rb') as wf:
        # initialize portaudio
        p = pyaudio.PyAudio()

        chirp = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    start=False)

        mic1 = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=mic1_id,
                        frames_per_buffer=chunk,
                        start=False)

        mic2 = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=mic2_id,
                        frames_per_buffer=chunk,
                        start=False)

        mic3 = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        input_device_index=mic3_id,
                        frames_per_buffer=chunk,
                        start=False)

        arr_mic1 = []
        arr_mic2 = []
        arr_mic3 = []

        mic1.start_stream()
        mic2.start_stream()
        mic3.start_stream()
        chirp.start_stream()

        for i in range(0, int(rate / chunk * record_seconds)):
            if len(data_chirp := wf.readframes(chunk)):
                chirp.write(data_chirp)
            else:
                chirp.stop_stream()

            data_mic1 = mic1.read(chunk)
            numpydata_mic1 = np.frombuffer(data_mic1, dtype=np.float32)
            arr_mic1.append(numpydata_mic1)

            data_mic2 = mic2.read(chunk)
            numpydata_mic2 = np.frombuffer(data_mic2, dtype=np.float32)
            arr_mic2.append(numpydata_mic2)

            data_mic3 = mic3.read(chunk)
            numpydata_mic3 = np.frombuffer(data_mic3, dtype=np.float32)
            arr_mic3.append(numpydata_mic3)


        # Close stream 
        chirp.close()

        mic1.stop_stream()
        mic1.close()

        mic2.stop_stream()
        mic2.close()

        mic3.stop_stream()
        mic3.close()

        p.terminate()

        # Process data
        signal_mic1 = np.ndarray.flatten(np.array(arr_mic1))
        signal_mic2 = np.ndarray.flatten(np.array(arr_mic2))
        signal_mic3 = np.ndarray.flatten(np.array(arr_mic3))

        # Sync based on first large amplitude event
        if sync_signal:
            trim_length = int(saved_seconds * rate)

            sync_threshold = 0.8

            mic1_max = np.max(signal_mic1)
            mic1_trim_ind = np.argmax(signal_mic1 > mic1_max * sync_threshold)
            signal_mic1 = signal_mic1[mic1_trim_ind-frame_shift:mic1_trim_ind-frame_shift+trim_length] 
            
            mic2_max = np.max(signal_mic2)
            mic2_trim_ind = np.argmax(signal_mic2 > mic2_max * sync_threshold) 
            signal_mic2 = signal_mic2[mic2_trim_ind-frame_shift:mic2_trim_ind-frame_shift+trim_length] 
            
            mic3_max = np.max(signal_mic3)
            mic3_trim_ind = np.argmax(signal_mic3 > mic3_max * sync_threshold)
            signal_mic3 = signal_mic3[mic3_trim_ind-frame_shift:mic3_trim_ind-frame_shift+trim_length] 

            plot_duration = saved_seconds
        else:
            plot_duration = record_seconds

        npy_arr = np.stack([signal_mic1, signal_mic2, signal_mic3])
        
        # Save data (if applicable)
        if saveData:
            nparr_str = "./Data/npy/" + file_id + "-arr.npy"
            np.save(nparr_str, npy_arr)

            if debug==True:
                wav_dir = "./Data/wav/" + file_id + "/"
                Path(wav_dir).mkdir(parents=True, exist_ok=True)

                write(wav_dir + "mic1.wav", rate, signal_mic1)
                write(wav_dir + "mic2.wav", rate, signal_mic2)
                write(wav_dir + "mic3.wav", rate, signal_mic3)

                # plot data
                t = np.linspace(0, plot_duration*1000, num=np.shape(signal_mic1)[0], endpoint=False)
                fig, axs = plt.subplots(3)
                fig.suptitle('Recorded Wave')
                axs[0].plot(t, signal_mic1)
                axs[1].plot(t, signal_mic2)
                axs[2].plot(t, signal_mic3)

                axs[0].set_title("Mic1")
                axs[1].set_title("Mic2")
                axs[2].set_title("Mic3")

                for ax in axs.flat:
                    ax.set(xlabel='Time(ms)', ylabel='Amplitude')

                plt.savefig("./Data/Plots/" + file_id + ".png")
                plt.close()
            
    return npy_arr