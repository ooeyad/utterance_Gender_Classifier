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
  - 1D CNN : The final shape of the matrix is (number_of_samples, (40 * 87) ,1) and the structure is shown gender_classifier.ipynb notebook.
  - 2D CNN : The final shape of the matrix is (number_of_samples, 40 , 87 ,1) and the structure is shown in the gender_classifier.ipynb  notebook.
6- An early sopping, learning rate schedule and checkpoint were added to the callbacks list, the best model is saved automatically.

## Approaches comparison
from the table "Comparisons.docx" the comparison shows that the best model is Approach 2 (model 2 (2D) )
as it has the best F1 Score from all, plus it has fewer trainable paremeters.

So I select Model_2 for the solution to build.

all tasks including : data extract, data transformation, exploratory data analysis, model training and experiments were done in gender_classifier.ipynb jupyter Notebook in google colab.

# How to use gender_classifier
- load the model files mfcc_model_2.h5, mfcc_model_2.json into a specific location that is accessible to your program.
- import gender_classifier.py
- instantiate the class by passing the directory where both files mfcc_model_2.h5, mfcc_model_2.json are located.
- call predict_gender and pass any audio file path of an utterance.
- it will return a single value indicating the gender of the speaker (M: for Male, F: for Female)
Example : 

from gender_classifier import Speakers_Gender_Classifier
clf = Speakers_Gender_Classifier("/content")
file_path = '/content/Recording_5.m4a'
clf.predict_gender(file_path)
