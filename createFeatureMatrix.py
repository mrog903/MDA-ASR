import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.io import wavfile as wav
import os

#Use this script to create and store a feature matrix for training ML model

#number of mfcc features to extract from audio samples
num_mfcc_features = 20

# Load dataset
metadata = pd.read_csv('C:/Users/max.rogers/Documents/UrbanSound8K/metadata/UrbanSound8K.csv')
fulldatasetpath = 'C:/Users/max.rogers/Documents/UrbanSound8K/audio'

# Create a list of the class labels
labels = list(metadata['class'].unique())

#function to extract features from audio file
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc_features)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

features = []

# Iterate through each sound file and extract the features
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_features(file_name)
    features.append([data, class_label])
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
#save as csv for future use
featuresdf.to_csv('C:/Users/max.rogers/Documents/UrbanSound8K/features/features_MFCC_{}.csv'
                  .format(num_mfcc_features), sep=',', header=None)
#save as pkl format for future use
featuresdf.to_pickle('C:/Users/max.rogers/Documents/UrbanSound8K/features/features_MFCC_{}.pkl'
                     .format(num_mfcc_features))
