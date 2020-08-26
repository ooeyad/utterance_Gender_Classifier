# Utterance_Gender_Classifier
A machine learning model to classify the gender of the speaker by analyzing utterances signals for different speakers.

## Data Collection
Data were crawled from "Open Speech and Language Resources" website http://www.openslr.org, the dataset of utterance audio consists of multiple speakers of large-scale (1000 hours) corpus of read English speech.
I have selected the clear speach part with the URL : http://www.openslr.org/resources/12/train-clean-100.tar.gz
- train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )

## About resource:

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

## Exploratory Data Analysis
Data contains SPEAKERS.TXT file that list all spekeakers ids labeled with Gender with additional information as total speaking minutes.

I grapped information of the speakers using SPEAKERS.TXT and selected arbitrary 40 files per speaker.

data are balanced for speakers' Genders so that would be helpful for the classification and no need for balancing the data

I have done some Data Augmentation to add more variations on data by taking random 1% of the audio files and added randomly colored noise and white noise and a slight pitch changes.

I have built two solutions to classify speakers genders from audio files :
1- features statistical analysis with deep neural network
2- MFCC matrix with Convolution Neural Network.

before building the models I have extracted the features for both solutions and saved them in the drive, so they will be ready to use for any changes in the networks.

# Classifier solution Approaches

## Approach 1 : Features Statistical Analysis with a simple Deep Neural Network.

In this approach I've implemented the following steps :
1- Prepared data files paths with speakers ids and genders in a dataframe
2- from the dataframe I loaded each audio file for 2 seconds only.
3- extracted the features from the audio signal and calculate mean, minimum, maximum and standard deviation for such feature
4- features extracted are :
*   Zero crossing rates : A measure of number of times in a given time interval/frame that the amplitude of the speech signals passes through a value of zero.
*   Tonal centroid : The Tonal Centroids (or Tonnetz) contain harmonic content of a given audio signal.
*   Roll-off frequency : The rate at which attenuation increases beyond the cut-off frequency
*   Spectral flattness : A measure to quantify how much noise-like a sound is, as opposed to being tone-like
*   Spectral Contrast : The level difference between peaks and valleys in the spectrum
*   Spectral Bandwidth : The difference between the upper and lower frequencies in a continuous band of frequencies.
*   Spectral centroid : Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.
*   RMSE : root-mean-square (RMS) energy for each frame from the audio sample.

5- So from all these features there are hidden information that could help classifying the signals information.
6- I have built an XGboost classifier and train it on the scaled dataset and retrieved the features importance.
7- 

## Approach 2 : MFCC features with Convolutional Neural Network.

In this approach I I've implemented the following steps :
1- Prepared data files paths with speakers ids and genders in a dataframe
2- from the dataframe I loaded each audio file for 2 seconds only.
3- Extraced the MFCC features for each audio, I have taken 40 mfccs and framed them to 87 frame.
5- I have built a CNN for two options : 1D and 2D 
  - 1D CNN : The final shape of the matrix is (number_of_samples, (40 * 87) ,1) and the structure is shown below notebook.
  - 2D CNN : The final shape of the matrix is (number_of_samples, 40 , 87 ,1) and the structure is shown in the gender_classifier.ipynb  notebook.
6- An early sopping, learning rate schedule and checkpoint were added to the callbacks list, the best model is saved automatically.

Model: "Model 1 (1D CNN)"
_________________________________________________________________
Layer (type)                 Output Shape              Paramnum 
=================================================================
conv1d_32 (Conv1D)           (None, 3480, 16)          112       
_________________________________________________________________
max_pooling1d_32 (MaxPooling (None, 1740, 16)          0         
_________________________________________________________________
conv1d_33 (Conv1D)           (None, 1740, 8)           776       
_________________________________________________________________
max_pooling1d_33 (MaxPooling (None, 870, 8)            0         
_________________________________________________________________
conv1d_34 (Conv1D)           (None, 870, 16)           784       
_________________________________________________________________
max_pooling1d_34 (MaxPooling (None, 435, 16)           0         
_________________________________________________________________
conv1d_35 (Conv1D)           (None, 435, 8)            776       
_________________________________________________________________
max_pooling1d_35 (MaxPooling (None, 218, 8)            0         
_________________________________________________________________
time_distributed_8 (TimeDist (None, 218, 8)            0         
_________________________________________________________________
dropout_38 (Dropout)         (None, 218, 8)            0         
_________________________________________________________________
flatten_24 (Flatten)         (None, 1744)              0         
_________________________________________________________________
dense_61 (Dense)             (None, 256)               446720    
_________________________________________________________________
output_layer (Dense)         (None, 1)                 257       
=================================================================
Total params: 449,425
Trainable params: 449,425
Non-trainable params: 0



Model: "Model 2 (2D CNN)"
_________________________________________________________________
Layer (type)                 Output Shape              Paramnum  
=================================================================
conv2d_22 (Conv2D)           (None, 35, 82, 16)        592       
_________________________________________________________________
max_pooling2d_22 (MaxPooling (None, 18, 41, 16)        0         
_________________________________________________________________
batch_normalization_22 (Batc (None, 18, 41, 16)        64        
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 13, 36, 16)        9232      
_________________________________________________________________
max_pooling2d_23 (MaxPooling (None, 7, 18, 16)         0         
_________________________________________________________________
batch_normalization_23 (Batc (None, 7, 18, 16)         64        
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 5, 16, 8)          1160      
_________________________________________________________________
max_pooling2d_24 (MaxPooling (None, 3, 8, 8)           0         
_________________________________________________________________
batch_normalization_24 (Batc (None, 3, 8, 8)           32        
_________________________________________________________________
flatten_22 (Flatten)         (None, 192)               0         
_________________________________________________________________
dropout_36 (Dropout)         (None, 192)               0         
_________________________________________________________________
dense_60 (Dense)             (None, 256)               49408     
_________________________________________________________________
dropout_37 (Dropout)         (None, 256)               0         
_________________________________________________________________
output_layer (Dense)         (None, 1)                 257       
=================================================================
Total params: 60,809
Trainable params: 60,729
Non-trainable params: 80
