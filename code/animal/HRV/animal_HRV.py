#split dataset to files 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import numpy as np
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import csv
from csv import DictWriter
from sklearn import preprocessing
import heartpy as hp
import scipy
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features
from hrvanalysis import plot_psd
from hrvanalysis import plot_distrib,plot_timeseries
from hrvanalysis import plot_timeseries


#for dog subjects
#data = loadmat("elctrography_Dog_01.mat")

#for rabbit subjects
#data = loadmat("elctrography_Rabbit_01_part_1.mat")

#for mouse subjects
data = loadmat("elctrography_Mouse_01.mat")

ecg= data['Data']
fs = data['Fs'].item(0)
time=math.floor(len(ecg)/fs)
points=time*256

print("freq",fs)
print("ecg initial len",len(ecg))
print("ecg time",time)
print("ecg new points",points)


#downsampling
resampled_signal = scipy.signal.resample( ecg, points)

print("ecg resampled len",len(resampled_signal))


#r peak detector
detectors = Detectors(256)
r_peaks = detectors.engzee_detector(resampled_signal[0:34304])
rr = np.diff(r_peaks)


'''
#r peak plot
print(r_peaks)

plt.figure()
plt.plot(ecg[0:2560])
plt.plot(r_peaks, ecg[r_peaks], 'ro')
plt.title('Detected R-peaks')
plt.savefig('new_downsampled_rpeaks.png', dpi=300)   #plot for 5 seconds (2500 points)
'''

#HRV time domain parameters
time_domain_features = get_time_domain_features(r_peaks)
print(time_domain_features )


#HRV frequency domain parameters
plot_psd(rr , method="welch")
plt.show()
