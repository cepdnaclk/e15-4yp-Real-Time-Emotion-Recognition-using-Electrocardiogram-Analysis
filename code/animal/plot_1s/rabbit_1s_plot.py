#plot rabbit subject ECG
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np
import csv
from csv import DictWriter
from sklearn import preprocessing
import scipy
from scipy import signal
import math
import pathlib
from ecgdetectors import Detectors

#load rabbit subject ecg data
rabbit_data = loadmat("elctrography_Rabbit_01_part_1.mat")
ecg= rabbit_data['Data']

#frequency
fs = rabbit_data['Fs'].item(0)

channel1= rabbit_data['Channels'].item(0)

print("Freq",fs)
print("Channel 1 ",channel1)
print("ECG ",ecg)


time=math.floor(len(ecg)/fs)
points=time*256

#print("ecg initial len",len(ecg))
print("ecg time",time)
#print("ecg new points",points)

#downsampling
resampled_signal = scipy.signal.resample( ecg, points)

print("ecg resampled len",len(resampled_signal))

#plot for 1 seconds (256 points)
plt.figure()
plt.plot(ecg[0:256])
plt.title("ECG signal for 1s - Rabbit Subject")
plt.show()

