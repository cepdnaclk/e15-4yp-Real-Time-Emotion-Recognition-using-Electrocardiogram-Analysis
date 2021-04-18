from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
from keras.utils.vis_utils import plot_model
import multiprocessing as mp
from keras.callbacks import History 
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU

import keras
import keras.backend as K

cce = tf.keras.losses.CategoricalCrossentropy()
sce = tf.keras.losses.SparseCategoricalCrossentropy()

adam = keras.optimizers.Adam(lr=0.0001)
SGD = keras.optimizers.SGD(lr=0.0001)

def soft_acc(y_true, y_pred):
  return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

import pandas as pd 
import random
import os
import csv
import numpy as np

from dataGen_classify import DataGenerator 

count=0
mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

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

cnn.add(layers.Conv1D(filters = 64, kernel_size = (10), input_shape = (853, 13)))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.MaxPooling1D(pool_size = (6)))
cnn.add(layers.Dropout(0.2))

cnn.add(layers.Conv1D(filters = 32, kernel_size = (2)))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.MaxPooling1D(pool_size = (3)))

#cnn.add(layers.Conv1D(filters = 6, kernel_size = (2), activation = 'relu'))
#cnn.add(layers.MaxPooling1D(pool_size = (2)))

cnn.add(layers.Flatten())

#cnn.add(layers.Dense(300, activation = "relu"))
#cnn.add(layers.Dense(200, activation = "relu"))
#cnn.add(layers.Dense(120, activation = "relu"))
cnn.add(layers.Dense(200))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.Dense(300))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.Dense(300))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.Dense(100))
cnn.add(layers.LeakyReLU(alpha=0.1))

cnn.add(layers.Dense(5, activation = "softmax"))
cnn.summary()
#plot_model(cnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




cnn.compile(optimizer= adam, loss=cce, metrics=['categorical_accuracy'])

history = cnn.fit(x = training_generator,validation_data = validation_generator, steps_per_epoch = 111, validation_steps = 32, epochs = 1000)
#history = cnn.fit(x = training_generator,validation_data = validation_generator, steps_per_epoch = 1, validation_steps = 1, epochs = 2)

# list all data in history
print(history.history.keys())


# summarize history for accuracy
#plt.plot(history.history['accuracy'])
# plt.plot(history.history['categorical_accuracy'])
# plt.plot(history.history['val_categorical_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('plots1.png', dpi=300)# summarize history for loss

fig, axs = plt.subplots(2)
#fig.suptitle('Model Loss and Accuracy')

axs[0].plot(history.history['categorical_accuracy'])
axs[0].plot(history.history['val_categorical_accuracy'])
axs[0].set(ylabel = 'Accuracy')
axs[0].legend(['train', 'test'], loc='upper left')
axs[0].set_title("Model Accuracy")


axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set(xlabel= 'Epoch', ylabel = 'Loss')
axs[1].legend(['train', 'test'], loc='upper left')
axs[1].set_title("Model Loss")


plt.savefig('plots1.png', dpi=300)# summarize history for loss



#plt.plot(history.history['loss'])
#plt.plot(history.history['val_mean_squared_error'])
#plt.title('Model Loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('plots3.png', dpi=300)

#for layer in cnn.layers: print(layer.get_weights())

# pred_val = cnn.predict(test_generator)
# print("Predictions")
# print(pred_val)

test_loss, test_acc = cnn.evaluate(test_generator, batch_size=512)

print("Test Loss : ", test_loss)
print("Test Acc : ", test_acc)



# for val in range(0, 400):
#   print(pred_val[val], test_y[val])

# print(history.history['mean_absolute_error'])

# print(cnn.input)
# print(cnn.output)

cnn.save('./MyModel_tf_new_1',save_format='tf')
cnn.save_weights('classify_new_1.h5')