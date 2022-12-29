# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:15:05 2022

@author: Antonio
"""

import numpy as np
from joblib import Parallel, delayed
import joblib
import tensorflow as tf
import librosa
from recording_helper import record_audio, terminate, get_features, create_wav_file

# !! Modify this in the correct order
commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']

loaded_model = tf.keras.models.load_model('my_model.h5')

def predict_mic():
    samples, rate = librosa.load(create_wav_file(), sr=8000)
    prediction = loaded_model.predict([[np.array(samples).reshape(-1,8000,1)]])
    prediction = np.argmax(prediction)
    command = commands[prediction]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        print(command)
        if command == "stop":
            terminate()
            break

predict_mic()