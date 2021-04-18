#HRV parametes of Humans
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
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features
from hrvanalysis import plot_psd
from hrvanalysis import plot_distrib,plot_timeseries
from hrvanalysis import plot_timeseries


#load dataset for humans
dreamer_data= loadmat('DREAMER.mat')

#take one signal 
signal1 = pd.DataFrame(dreamer_data['DREAMER'].item(0,0)[0].item(2).item(0)[3]['stimuli'].item(0)[5][0],columns=["c1","c2"] )

#take one channel
c11=signal1['c1']


#r peak detector
detectors = Detectors(256)
r_peaks = detectors.engzee_detector(c11[0:34304])
rr = np.diff(r_peaks)

#HRV time domain parameters
time_domain_features = get_time_domain_features(r_peaks)
print(time_domain_features )


#HRV frequency domain parameters
plot_psd(rr , method="welch")
plt.show()
