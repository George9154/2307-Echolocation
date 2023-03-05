import time
import wave
import pyaudio
import numpy as np
from matplotlib import pyplot as plt

from scipy.io.wavfile import write
from pathlib import Path


def checkInputDevices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


def checkFolders():
    Path("./Data/").mkdir(parents=True, exist_ok=True)
    Path("./Data/wav/").mkdir(parents=True, exist_ok=True)
    Path("./Data/npy/").mkdir(parents=True, exist_ok=True)
    Path("./Data/Plots/").mkdir(parents=True, exist_ok=True)


def record(mics, savewav=False, plotting=False):
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    CHUNK = 420 # fixed chunk size

    RECORD_SECONDS = 5

    mic1_id = mics[0]
    mic2_id = mics[1]
    mic3_id = mics[2]

    with wave.open("./chirp3ms.wav", 'rb') as wf:
        # initialize portaudio
        p = pyaudio.PyAudio()

        chirp = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

        mic1 = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=mic1_id,
                        frames_per_buffer=CHUNK)

        mic2 = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=mic2_id,
                        frames_per_buffer=CHUNK)

        mic3 = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=mic3_id,
                        frames_per_buffer=CHUNK)

        arr_mic1 = []
        arr_mic2 = []
        arr_mic3 = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            
            data_chirp =  wf.readframes(CHUNK)
            if len(data_chirp):
                chirp.write(data_chirp)

            data_mic1 = mic1.read(CHUNK)
            numpydata_mic1 = np.frombuffer(data_mic1, dtype=np.float32)
            arr_mic1.append(numpydata_mic1)

            data_mic2 = mic2.read(CHUNK)
            numpydata_mic2 = np.frombuffer(data_mic2, dtype=np.float32)
            arr_mic2.append(numpydata_mic2)

            data_mic3 = mic3.read(CHUNK)
            numpydata_mic3 = np.frombuffer(data_mic3, dtype=np.float32)
            arr_mic3.append(numpydata_mic3)

        # Close stream 
        chirp.stop_stream()
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

        # Save data
        npy_arr = np.stack([signal_mic1, signal_mic2, signal_mic3])

        # Data sample ID: current time
        curr_time = time.strftime("%Y%m%d-%H%M%S")
        
        nparr_str = "./Data/npy/" + curr_time + "-arr.npy"
        np.save(nparr_str, npy_arr)

        if savewav==True:
            wav_dir = "./Data/wav/" + curr_time + "/"
            Path(wav_dir).mkdir(parents=True, exist_ok=True)

            write(wav_dir + "mic1.wav", RATE, signal_mic1)
            write(wav_dir + "mic2.wav", RATE, signal_mic2)
            write(wav_dir + "mic3.wav", RATE, signal_mic3)

        if plotting==True:
            # plot data
            t = np.linspace(0, RECORD_SECONDS*1000, num=np.shape(signal_mic1)[0], endpoint=False)
            fig, axs = plt.subplots(3)
            fig.suptitle('Recorded Wave')
            axs[0].plot(t, signal_mic1)
            axs[1].plot(t, signal_mic2)
            axs[2].plot(t, signal_mic3)

            for ax in axs.flat:
                ax.set(xlabel='Time(ms)', ylabel='Amplitude')

            plt.savefig("./Data/Plots/" + curr_time + ".png")



if __name__ == "__main__":
    checkInputDevices()
    checkFolders()

    mics = [0,1,2]
    record(mics, savewav=True, plotting=True)