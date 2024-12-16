import pandas as pd
import os

SEARCH_TERMS = ['rain', 'windy', 'hail', 'thunder', 'snow']
OUTPUT_DIR = 'weather_audio_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = []
for term in SEARCH_TERMS:
    for file in os.listdir(OUTPUT_DIR):
        if file.startswith(term):
            metadata.append({'file_name': file, 'label': term})

pd.DataFrame(metadata).to_csv('data/audio_labels.csv', index=False, header=False)

# # Load the CSV file
# df = pd.read_csv('audio_labels.csv', header=None, names=['file_name', 'label'])
#
# # Count the number of files for each label
# label_counts = df['label'].value_counts()
#
# # Display the counts
# print(label_counts)