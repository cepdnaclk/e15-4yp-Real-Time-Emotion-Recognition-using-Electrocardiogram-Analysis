from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import pandas as pd 
import random
import os
import csv
import numpy as np

import keras
import keras.backend as K

count=0


cce = tf.keras.losses.CategoricalCrossentropy()
sce = tf.keras.losses.SparseCategoricalCrossentropy()

adam = keras.optimizers.Adam(lr=0.0001)
SGD = keras.optimizers.SGD(lr=0.0001)


data_path='data'
label_path='labels1'

partition={
    'train':[],
    'validation':[],
    'test':[]  
}


params = {'dim': (853),
          'batch_size': 512,
          'n_classes': 5,
          'n_channels': 13,
          'shuffle': True}


data_dir_list = os.listdir(data_path)
label_dir_list = os.listdir(label_path)

print(len(data_dir_list))
print(len(label_dir_list))

train_arr=[]
valid_arr=[]
test_arr=[]
temp_arr=[]

num_signal_files = len(data_dir_list)

num_signals_temp = (num_signal_files*30)//100


label_name_arr=[]

#randomly selcet test signals
temp_list_index = random.sample(range(0, num_signal_files-1), num_signals_temp)


for i in range(1,num_signal_files+1):
    signal_filename='data/signal-'+str(i)
    
    
    count=count+1
    #if(i<414):
 		

    if i in temp_list_index:
        temp_arr.append(signal_filename)
        
    else:
        train_arr.append(signal_filename)
        
num_signals_test = (num_signals_temp*33)//100

test_list_index = random.sample(range(0, num_signals_temp-1), num_signals_test)
             
for j in range(1, num_signals_temp+1):

    signal_filename='data/signal-'+str(j)
    
    if j in test_list_index:
        test_arr.append(signal_filename)
        
    else:
        valid_arr.append(signal_filename)
    


partition['train']=train_arr
partition['validation']=valid_arr
partition['test']=test_arr


#print('Train length', len(train_arr))
#print('Valid length', len(valid_arr))


label_arr=[]

num_label_files = len(label_dir_list)
for x in range(1,num_label_files+1):
    
    label_filename='data/signal-'+str(x)
    label_name_arr.append(label_filename)

    #label=np.load(label_filename+'.npy',allow_pickle=True) 

    #label_arr.append(label)
    
    
#print(label_arr) 
#print(count)
      
   


labels = dict.fromkeys(label_name_arr)
#print(label)


for y in range(1,num_label_files+1):
    
    label_filename='labels1/signal-'+str(y)
    #label_name_arr.append(label_filename)

    label=np.load(label_filename+'.npy',allow_pickle=True) 

    label_arr.append(label)
    
    
#print(label_arr) 

i=0
for key in labels :
	labels[key]=label_arr[i]
	#print(labels[key])
	#print(labels[key])
	i=i+1


print('hi1')

# Datasets
# partition = # IDs
# labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
print('hi2')
validation_generator = DataGenerator(partition['validation'], labels, **params)

test_generator= DataGenerator(partition['test'], labels, **params)

#print(type(training_generator))
#print(type(validation_generator))

#print(training_generator.__len__())
#print(validation_generator.__len__())

# print(x_trian.shape)
# print(type(x_train))

# print(y_trian.shape)
# print(type(y_train))

# print(x_test.shape)
# print(type(x_test))

# print(y_test.shape)
# print(type(y_test))

#print(training_generator.__data_generation())
print('hi3')
#print('Tuple train', np.shape(tuple(training_generator)))
#print('Tuple valid', np.shape(tuple(validation_generator)))


print('hi4')

#print(test_generator.__getitem__(0)[0])
#print(test_generator.__getitem__(0)[1])

# only the first batch
test_y = test_generator.__getitem__(0)[1]




cnn = models.Sequential()
cnn.add(LSTM(100))
cnn.add(layers.Dense(1, activation = 'softmax'))
cnn.summary()

#cnn.compile(optimizer= adam, loss=cce, metrics=['categorical_accuracy'])

cnn.compile(optimizer= adam, loss=cce, metrics=['categorical_accuracy'])

history = cnn.fit(x = training_generator,validation_data = validation_generator, steps_per_epoch = 111, validation_steps = 32, epochs = 1000)
#history = cnn.fit(x = training_generator,validation_data = validation_generator, steps_per_epoch = 1, validation_steps = 1, epochs = 2)

# list all data in history
print(history.history.keys())
test_loss, test_acc = cnn.evaluate(test_generator, batch_size=512)