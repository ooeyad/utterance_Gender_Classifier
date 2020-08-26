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


# Classifier solution Approaches

## Approach 1 : Features Statistical Analysis.

In this approach I do the following :
1- Prepare data files paths with speakers ids and genders in a dataframe
2- from the dataframe load each audio file for 2 seconds only.
3- extract the features from the audio signal and calculate mean, minimum, maximum and standard deviation for such feature
4- features extracted are :
*   zero crossing rates
*   tonal centroid
*   roll-off frequency
*   spectral flattness
*   Spectral Contrast
*   Spectral Bandwidth
*   spectral centroid
*   RMSE
