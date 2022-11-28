from typing import Dict, List, Tuple

import pyaudio
import numpy as np
import time
import wave
import datetime
import os


def pyserial_start() -> Tuple[pyaudio.Stream, pyaudio.PyAudio]:

    # -- streaming can be broken down as follows:
    # -- -- format             = bit depth of audio recording (16-bit is standard)
    # -- -- rate               = Sample Rate (44.1kHz, 48kHz, 96kHz)
    # -- -- channels           = channels to read (1-2, typically)
    # -- -- input_device_index = index of sound device
    # -- -- input              = True (let pyaudio know you want input)
    # -- -- frames_per_buffer  = chunk to grab and keep in buffer before reading

    audio = pyaudio.PyAudio()  # create pyaudio instantiation
    stream = audio.open(
        format=pyaudio_format,
        rate=samp_rate,
        channels=channel,
        input_device_index=dev_index,
        input=True,
        frames_per_buffer=chunk,
    )
    stream.stop_stream()  # stop stream to prevent overload
    return stream, audio


def data_grabber(rec_len: int) -> Tuple[List[np.ndarray], List[pyaudio.Stream.read], datetime.datetime]:
    stream.start_stream()  # start data stream
    stream.read(chunk, exception_on_overflow=False)  # flush port first
    t_0 = datetime.datetime.now()  # get datetime of recording start
    print("Recording started.")
    data = []
    data_frames = []

    for frame in range(0, int((samp_rate * rec_len) / chunk)):
        stream_data = stream.read(
            chunk, exception_on_overflow=False
        )  # grab data frames from buffer
        data_frames.append(stream_data)  # append data
        data.append(np.frombuffer(stream_data, dtype=buffer_format))
    stream.stop_stream()  # stop data stream
    print("Recording Stopped.")
    return data, data_frames, t_0


def data_saver(t_0: datetime.datetime) -> str:
    data_folder = "/home/pi/System_monitorowania/audio/"  # folder where data will be saved locally
    if os.path.isdir(data_folder) == False:
        os.mkdir(data_folder)  # create folder if it doesn't exist
    filename = datetime.datetime.strftime(
        t_0, "%Y_%m_%d_%H_%M_%S_pyaudio"
    )  # filename based on recording time
    wf = wave.open(data_folder + filename + ".wav", "wb")  # open .wav file for saving
    wf.setnchannels(channel)  # set channels in .wav file
    wf.setsampwidth(audio.get_sample_size(pyaudio_format))  # set bit depth in .wav file
    wf.setframerate(samp_rate)  # set sample rate in .wav file
    wf.writeframes(b"".join(data_frames))  # write frames in .wav file
    wf.close()  # close .wav file
    return filename


if __name__ == "__main__":

    # acquisition parameters
    chunk = 44100  # frames to keep in buffer between reads
    samp_rate = 44100  # sample rate [Hz]
    pyaudio_format = pyaudio.paInt24  # 16-bit device
    buffer_format = np.int16  # 16-bit for buffer
    channel = 1  # only read 1 channel
    dev_index = 2  # index of sound device
    # stream info and data saver

    while True:
        stream, audio = pyserial_start()  # start the pyaudio stream
        record_length = 1  # seconds to record
        data_chunks_all, data_frames, t_0 = data_grabber(record_length)  # grab the data
        data_saver(t_0)  # save the data as a .wav file
        time.sleep(4)  # pause recording for 4 seconds
        # pyserial_end()  # close the stream/pyaudio connection
