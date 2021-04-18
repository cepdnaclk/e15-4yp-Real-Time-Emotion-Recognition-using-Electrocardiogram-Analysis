import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from csv import DictWriter
from sklearn import preprocessing
import scipy
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors

#load mouse subject ecg data
mouse_data = loadmat("elctrography_Mouse_01.mat")
ecg= mouse_data['Data']

#frequency
fs = mouse_data['Fs'].item(0)

channel1= mouse_data['Channels'].item(0)

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
plt.title("ECG signal for 1s - Mouse Subject")
plt.show()

