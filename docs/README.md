---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e15-4yp-Real-Time-Emotion-Recognition-using-Electrocardiogram-Analysis
title: Real Time Emotion Recognition using Electrocardiogram Analysis
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Project Title

#### Team

- E/15/139, Ishanthi D.S. , [e15139@eng.pdn.ac.lk](mailto:e15139@eng.pdn.ac.lk)
- E/15/249, Pamoda W.A.D. , [dasunip2@gmail.com](mailto:dasunip2@gmail.com)
- E/15/299, Ranushka L.M. , [e15299@eng.pdn.ac.lk](mailto:e15299@eng.pdn.ac.lk)

#### Supervisors

- Dr. Isuru Nawinne, [isurunawinne@eng.pdn.ac.lk](mailto:isurunawinne@eng.pdn.ac.lk)
- Prof. Roshan Ragel, [roshanr@eng.pdn.ac.lk](mailto:roshanr@eng.pdn.ac.lk)
- Dr. Suranji Wijekoon, [suranjisk@gmail.com](mailto:suranjisk@gmail.com)
- Mr. Theekshana Dissanayake, [theekshanadis@eng.pdn.ac.lk](mailto:theekshanadis@eng.pdn.ac.lk)  

#### Table of content

1. [Abstract](#abstract)
2. [Methodology](#methodology)
3. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
4. [Results and Analysis](#results-and-analysis)
5. [Conclusion](#conclusion)
6. [Links](#links)

---

## Abstract
Most of the ECG analysis based human emotion recognition studies use different types of machine learning techniques. Main problem with these methods is lack of accuracy and not having the ability to classify emotions real-time. The proposed method uses a large public dataset to increase accuracy and implements a Convolutional Neural Network to identify emotions. ECG data signals are preprocessed to increase the number of instances and important features are extracted using feature extraction methods and then features are fed to the CNN. Three CNN models are trained to predict the valence, arousal and the dominance values of the ECG signal, which are used to finalize the emotion by mapping those values to the valence-arousal-dominance 3D plane.  The classification CNN models implemented in this proposed method result in a maximum accuracy of 80%.



## Methodology
This research is planned to do in 2 phases to implement the human emotion recognition model and to do a preliminary analysis on animal emotion recognition.
Phase 1 - Human Emotion Recognition
As the first phase of the research, human emotion recognition model was implemented using ECG signals. The bio signals such as ECG signals are nonlinear, complex and contain noises. Since neural networks handle such data in a more efficient way than machine learning methods, neural network is a more suitable method for the classification of emotions using ECG signals.
In neural networks, convolutional neural networks (CNN) are more suitable for time series analyzing since they identify and extract important features that support the classification process from raw input data. Therefore, the model for the recognition of human emotions using ECG signals was implemented using CNNs.

For the human emotion recognition model, a public dataset, which consists of ECG signal data of human subjects was used. These data signals are preprocessed to increase the number of instances and to extract the important features to feed to the CNN. The CNN is trained to predict the valence, arousal and the dominance values of the ECG signal, which is used to finalize the emotion by mapping those values on the valence-arousal-dominance 3D plane.



## Experiment Setup and Implementation
Dataset
For the training of a neural network, a large amount of ECG signal data was needed. Therefore, the DREAMER dataset, which is a multi-model dataset for emotion recognition through ECG and EEG signals, was used for this research.
This dataset consists of the ECG signals of 23 human subjects, each subject’s ECG signals have been recorded using 2 ECG channels under 256 Hz sampling rate while watching 18 video clips. They have been asked to score their emotions using valence, arousal and dominance values in a range of 0 to 5. Therefore, the total number of 414 labeled signals were obtained from this dataset.

To get an initial idea of the ECG signals, signals with different valence-arousal-dominance values are visualized. The output graphs  indicate that the ECG signal changes over a 5 seconds period 8 different valence, arousal, dominance value combinations. Each combination of these values are related to a different emotion of the human subject. It is visible that the amplitude, shape and the pattern of the signal are different for different emotional states.

Preprocessing
To increase the number of instances fed to the CNN, some preprocessing steps were done on the above mentioned ECG data signals.
Therefore, as the first step, those signals were split into sub segments. The emotion changes of a subject can be identified within a period of 3 – 15 seconds. Considering this fact, the above 414 signals were split into segments of 10 seconds period considering 1 second overlapping time. This process created 82 018 input instances for the CNN.

Feature Extraction
To extract the unique features of these ECG signals, Mel frequency cepstral coefficients (MFCC) algorithm, which is an efficient technique for signal processing based on Discrete Fourier Transform (DFT), was used. This calculates a MFCC feature vector with coefficient values for the input signal.

Each signal consisted of 2 ECG channels therefore each segment of the signal also consisted of 2 ECG channels and the MFCC feature extraction was done for both channels.  13 MFCC coefficients were extracted and this calculated feature vectors of size (853 x 13) for each segment. Since these segments have 2 channels, the input instance shape was (853 x 13 x 2). Both the input instances and the array of valence, arousal and dominance values were normalized. Each normalized input instance and the relevant normalized valence, arousal and dominance values were stored in separate files before training the CNN.

CNN
Finally, 82018 data instances of shape (853 x 13 x 2) were fed to CNN to predict the values of valence, arousal and dominance.

The dataset that was used had 2 channels for the ECG signal. And as the MFCC feature extraction method creates a 2D output of coefficients of time and frequency variation, a 2D convolutional Neural Network is used for the classification of ECG time series data.

Neural Network structure:
            	Convolutional Neural Network
                            	Conv2d - Input
                            	MaxPooling2d
                            	Dropout – 20%
                            	Conv2d
                            	MaxPooling2d
                            	Conv2d
                            	MaxPooling2d
                            	Flatten
                            	Dense layer - 200 nodes with ReLu activation
                            	Dense layer - 150 nodes with ReLu activation
                                Dense layer - 75 nodes with ReLu activation
                                Dense layer - 25 nodes with ReLu activation
                            	Dense layer - 3nodes with ReLu activation
Convolution Layers:
            	Convolution is a linear operation and it is done in parallel in the Conv2D layers. Conv2D layers are used for feature Mapping. Each Emotion elicited ECG signals are transformed into a set of 2D Coefficients and those coefficients contain the patterns related to the emotion. Therefore, in the training phase the filters of the convolution layers are trained to map the features so that they will act as the feature detectors.

MaxPooling Layers:
            	Pooling layers are used to down sample the given input to extract the features in another position. MaxPooling is used to extract the most activated feature among several features. Therefore before a convolutional layer a MaxPooling layer is placed.

Dropout Layer:
            	Dropout layers are used to delete some trained neurons. This process is done to train a model more accurately. It is expected to have a better training when 20% of trained neurons are reset to initial state.

Flatten layer:
            	This layer is used to prepare the inputs to the dense input layer.
Dense layer:
            	These layers are used to classify the inputs into 3 classes. Final dense layer has 3 output neurons. One or more hidden dense layers are expected to be used until a better accuracy is gained.

3 types of CNNs using regression CNN models and classification CNN models were implemented as different approaches to predict valence, arousal and dominance values.

Single Regression CNN model
In this approach, a single regression CNN model was implemented to predict 3 values of valence, arousal and dominance at the same time. Three labels of valence arousal and dominance in the range of 1-5 were used. The output of each model will be three normalized values in the range of 0-1 which describes valence, arousal and dominance values accordingly.

Separate Regression CNN models
In this approach, 3 separate regression CNN models were implemented to predict valence, arousal and dominance. The labels of valence arousal and dominance in the range of 1-5 were normalized into the values of 0.00 , 0.25, 0.50, 0.75, 1.00. The output of each model will be a normalized value in the range of 0-1 which describes valence, arousal and dominance values accordingly.

Separate Classification CNN models
In this approach, 3 separate classification CNN models were implemented to predict valence, arousal and dominance. The labels of valence arousal and dominance in the range of 1-5 were one hot encoded. The output of each model will be a value in the range of 0-4 which describes valence, arousal and dominance values accordingly.

Discrete emotional model
To identify and label the emotions related to predicted arousal, valence and dominance values, 3D- valence-arousal-dominance-plane is used. 4 negative discrete emotions as angry, fear, unconcerned, sad and 4 positive emotions as happy, surprise, satisfied and protected can be classified using this model.

Pitfalls and workarounds
This dataset was a large dataset with 82 018 instances. Initially we tried to load the whole datasets at once, but it required a lot of processing. Therefore, instead of having one large dataset, we created single data files for each instance and used python data generators to load data to the CNN.
When training the neural network we faced an issue of insufficient memory and low speed in the machine. This problem was solved by handling the training of the CNN on the GPU of the kepler server.


Phase 2 - Preliminary Analysis of Animal ECG data

Heart rate variability (HRV) is the physiological event of the variation in the time interval between consecutive heartbeats in milliseconds using ECG signals. Normally heart rate variability measure rises when a person is engaged in a relaxing activity and it reduces when a person is under stress. Therefore, HRV can be used as a measurement to assess the emotions of a person by evaluating his or her autonomic nervous system. Therefore, for this preliminary study of animal ECG signals, HRV parameters were used. Typically, HRV is analysed using time domain, frequency domain and non linear metrics.

Background

Time-Domain Parameters

In this method, QRS complex is identified and the heart rate at any point in time or the intervals between successive normal complexes are determined and the following parameters are calculated.
Some of the most widely calculated time domain parameters are:
mean_nni: Mean time interval between two heartbeats, here normal heartbeats are considered.
sdnn: the standard deviation of all the NN intervals. It can be described as a total variability or total power.
sdsd: the standard deviation of the differences between successive NN intervals
nni_50: the number of pairs of successive NN intervals that differ by more than 50 ms in the entire recording
pnni_50: the percentage of successive intervals that differ by more than 50 ms (higher values indicate increased parasympathetic activity)
rmssd: the square root of the root mean square of the sum of all differences between successive NN intervals
median_nni : Median Absolute values of the successive differences between the RR-intervals.
range_nni: difference between the maximum and minimum nn_interval.

Frequency-Domain Parameters
In frequency domain analysis, frequency components of the ECG signals are obtained as VLF (Very Low Frequency – 0.00 Hz- 0.04 Hz), LF (Low Frequency– 0.04 Hz- 0.15 Hz) and HF (High Frequency– 0.15 Hz- 0.40 Hz). This is performed by decomposing RR intervals of an ECG signal using Fast Fourier Transformation (FFT).

Dataset
The PhysioZoo database consists of ECG signal recordings taken from multiple types of mammals such as dogs, rabbits, mice, etc. It has dog ECG recordings of an average length 05.31 (min:sec), rabbit ECG recordings of an average length 10.34 (min:sec) and mouse ECG recordings of an average length 29.44 (min:sec). It has recorded these ECG data at a sampling rate of 500Hz for dog subjects, 1000 Hz for mouse and rabbit subjects.

Methodological approach
For animal ECG analysis, ECG signal data of dog subjects, mouse subjects and rabbit subjects were used from the above mentioned publicly available dataset. Initially they were downsampled to a frequency of 256 Hz. Then the HRV parameters were extracted from 134 s duration of ECG signals using time domain and frequency domain methods using python hrv-analysis library.  Then these parameters were compared with the human HRV parameters.

Data visualization
Initially the data was visualized to get a basic idea of the QRS complex and the shape of ECG signal recordings of each animal type.


## Results and Analysis
The CNN was trained using  60000 instances and 20000 testing data. The results were taken by training the model changing its parameters as well as the hyper-parameters such as batch size, steps per epoch, epochs etc.

CNN for classification of Arousal. Valence and Dominance Values
The dataset consisted of 82000 samples of MFCC features of ECG signals with 5 labels for Arousal category. The dataset was divided as follows,

	Training Set - 70%		Validation Set - 20%		  Test Set - 10%

In each three figures above has two graphs representing Model accuracy and Loss. If the Accuracy graphs (Top graph) are considered the orange line which represents the validation set accuracy has come to an instance which is called a valley after 1000 epochs. This implies that the model accuracy will not considerably increase further. All the three models show this kind of variation.

On the other hand, If the start of the graphs were considered, there has been no change in accuracy or loss in the first 180 epochs approximately. The optimizer used for the models was Adam optimizer. In general, in the search for the solution for a Machine learning problem, the models come across with many solution points which are seemingly better. But only one of them is the best solution. That solution is also called the “Global Minimum”. At that point the loss is minimum.  The other solutions are called “Local Minimas”. Therefore, the model must be able to travel in the path of global minima for a successful result. Following figure (Figure: 03) describes the difference between the global and local minimas. Therefore, It is clear that there is one global minimum and there could be one or more local minimas.

Clearly, the model tries to find the path for global minima in the beginning of the training time. As a result there will not be any change in the accuracy and the loss during that time. This is performed by the optimizer. Therefore, better optimizers must be used according to the model. There are different types of optimizers and each one has different properties. Adam optimizer is better for all the cases. Therefore it is used here.
Finally, The loss variation of each graph shows another great fact. As you can see the validation loss (Orange line) started reducing when the model has found the global minima and after a long time it starts to increase. But the loss of the training set never decreases. This instance shows the overfitting of the model. As the training loss decreases the test loss or the validation loss increases. By the end of the 1000th epoch the model has been trained only for the training set. Therefore, the moment the test loss starts to increase can be used as the end of the training if the model is going to be used for ECG data recorded by other devices. Therefore the models may actually have lesser accuracy than mentioned here. For example, The model developed for Dominance may have an accuracy of 70% instead of 80.87%.



## Conclusion
Emotion recognition is a powerful and very useful technique in the modern world since it has a large  scale of uses in various areas. We can say a lot of research has been conducted on this  subject using different methods.  Some methods give higher accuracy but some do not. Scientists have  come up with new techniques to increase the accuracy by inventing new feature extraction methods,  classification methods, machine learning models and neural networks. Even though some methods give  higher accuracy they may be practically hard to use because of the non- wearable nature of the  hardware implementation. Since emotion recognition is a key feature of Human Computer Interaction, as the field grows  in sophistication people need easily usable methods. That is why the combined bio signal method is not  so used even though it has a good accuracy. In the near future by these emotion recognition methods the Human Computer Interaction would be  more effective and by that the productivity will increase in every computer using field including  healthcare, education, production industry, entertainment and automotive industry etc, making human  and animal lives better and easier.
The highest emotion classification accuracy obtained by this study is 80.87%. This was achieved by overfitting the model of that dataset. Although the model was overfitted the model showed considerable accuracy before it overfits. MFCC features of the ECG signal were extracted. That produced a 2D feature vector of time and frequency. However, different feature extraction methods should be tried to increase the accuracy.

## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e15-4yp-Real-Time-Emotion-Recognition-using-Electrocardiogram-Analysis)
- [Project Page](https://cepdnaclk.github.io/e15-4yp-Real-Time-Emotion-Recognition-using-Electrocardiogram-Analysis/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
