# **Audio-Based Weather Prediction**

## **Project Overview**  
This project focuses on predicting weather conditions based on audio data. Using an XGBoost classifier, the model is capable of classifying five types of weather conditions:

- **Hail**  
- **Rain**  
- **Windy**  
- **Thunder**  
- **Snow**  

The dataset for this project was collected from [Freesound](https://freesound.org/), which contains various audio recordings of natural weather sounds.

---

## **Key Features**  
1. **Audio Classification using XGBoost**  
   - The XGBoost model is trained on processed audio features to classify weather conditions.

2. **Real-Time Weather Prediction**  
   - The project integrates real-time audio capture using the **`sounddevice`** library. This allows the system to record audio directly from the environment and predict weather conditions on the fly.

3. **Dataset**  
   - Audio recordings of weather sounds were sourced from [Freesound](https://freesound.org/) and preprocessed to extract relevant audio features like MFCC (Mel-Frequency Cepstral Coefficients).

---

## **Technologies Used**  
- **XGBoost**: For building the classification model.  
- **Sounddevice**: For real-time audio recording.  
- **Scikit-learn**: For preprocessing and evaluation (e.g., standardization, label encoding, and metrics).  
- **Pandas & NumPy**: For data handling and processing.

---

## **How to Run the Project**  

### **1. Installation**  
First, clone the repository and install the required libraries:  
```bash
git clone <repository-link>
cd <project-folder>
pip install -r requirements.txt
```
### **2. Running the Model**  
To train the XGBoost model and evaluate its performance:
```bash
python xgboost_implementation.py
```
**What this script does:**
- Loads and preprocesses the dataset
- Trains the XGBoost classifier using cross-validation
- Saves the trained model and scaler for future predictions
- Displays the model's accuracy, confusion matrix, and classification report

### **3.  Real-Time Prediction**  
To predict the weather condition in real time using your microphone:
```bash
python real_time_audio_test.py
```
**What this script does:**
- Captures live audio using the sounddevice library
- Preprocesses the recorded audio to extract relevant features
- Applies the trained XGBoost model to predict the weather condition
- Displays the predicted weather condition in the terminal

## Dataset
The dataset is collected from [Freesound](https://freesound.org/)
 and contains audio recordings representing various weather conditions: hail, rain, windy, thunder, and snow.

## How It Works

1. **Feature Extraction**: Extracts MFCC (Mel Frequency Cepstral Coefficients) features from audio.
2. **Model Training**: Trains an XGBoost model using labeled weather audio data.
3. **Real-Time Prediction**: Captures live audio, processes it, and predicts the weather condition.

## Future Improvements
- Improve the accuracy of model.
- Expand dataset size for better generalization.
- Integrate additional weather conditions.
- Deploy the model as a real-time web or mobile application.
