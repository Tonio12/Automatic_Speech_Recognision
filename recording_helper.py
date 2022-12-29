# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:56:26 2022

@author: Antonio
"""

import pyaudio
import numpy as np
import wave
import librosa
from sklearn.preprocessing import StandardScaler


FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    # print("recording stopped")

    stream.stop_stream()
    stream.close()
    
    return np.frombuffer(b''.join(frames), dtype=np.float32)


def terminate():
    p.terminate()
    
def create_wav_file():
        sound_file = wave.open('myrecording.wav','wb')
        sound_file.setnchannels(CHANNELS)
        sound_file.setsampwidth(p.get_sample_size(FORMAT))
        sound_file.setframerate(RATE)
        sound_file.writeframes(record_audio())
        sound_file.close()
        return 'myrecording.wav'
    

def get_features():
    y, SAMPLE_RATE = librosa.load('myrecording.wav', sr=None)
    rms = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for value in mfcc:
        to_append += f' {np.mean(value)}'
    data = list(map(float, to_append.split(" ")))
    return data
    
create_wav_file()