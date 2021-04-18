
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

#load dataset
dreamer_data= loadmat('DREAMER.mat')

#take one signal 
signal1 = pd.DataFrame(dreamer_data['DREAMER'].item(0,0)[0].item(2).item(0)[3]['stimuli'].item(0)[5][0],columns=["c1","c2"] )

#take one channel
c11=signal1['c1']


#plot for 1 seconds (256 points) , convert to ms
plt.figure()
plt.plot(c11[0:256]/1000)
plt.title("ECG signal for 1s - Human Subject")
plt.show()

