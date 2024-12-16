# from pydub import AudioSegment
# import os
#
# # Directory containing the MP3 files
# input_directory = 'weather_audio_data'
# # Directory to save the converted WAV files
# output_directory = 'weather_audio_data_wav'
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_directory, exist_ok=True)
#
# def convert_mp3_to_wav(mp3_file, output_file):
#     audio = AudioSegment.from_mp3(mp3_file)
#     audio.export(output_file, format="wav")
#
# # Loop through all MP3 files in the input directory and convert them
# for file in os.listdir(input_directory):
#     if file.endswith('.mp3'):
#         mp3_file = os.path.join(input_directory, file)
#         wav_file = os.path.join(output_directory, file.replace('.mp3', '.wav'))
#         convert_mp3_to_wav(mp3_file, wav_file)
#         print(f"Converted {file} to WAV format")
#
# print("Conversion complete!")

# to model readable
import librosa
import numpy as np
import pandas as pd
import os

# Directory containing the WAV files
input_directory = 'weather_audio_data_wav'
# Output CSV file for storing extracted features
output_csv = 'audio_features.csv'

def extract_features(file_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        # Extract MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Compute mean and standard deviation across time for each coefficient
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        # Combine mean and standard deviation into a single feature vector
        feature_vector = np.hstack((mfccs_mean, mfccs_std))
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare a list to store features and labels
features = []
labels = []

# Loop through all WAV files and extract features
for file in os.listdir(input_directory):
    if file.endswith('.wav'):
        file_path = os.path.join(input_directory, file)
        label = file.split('_')[0]  # Extract label from file name
        feature_vector = extract_features(file_path)
        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(label)

# Convert features and labels to a DataFrame
columns = [f'mfcc_mean_{i}' for i in range(13)] + [f'mfcc_std_{i}' for i in range(13)]
df = pd.DataFrame(features, columns=columns)
df['label'] = labels

# Save the DataFrame to a CSV file
df.to_csv(output_csv, index=False)
print(f"Features saved to {output_csv}")

