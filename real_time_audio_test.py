import sounddevice as sd
import numpy as np
import joblib
import xgboost as xgb
import librosa

# Load the trained model and scaler
model = xgb.XGBClassifier()
model.load_model('weather_predictor_model.json')  # Load the saved model
scaler = joblib.load('model_save/scaler.pkl')  # Load the saved scaler

# Weather conditions mapping
weather_conditions = ["rain", "windy", "hail", "thunder", "snow"]

# Feature extraction function (only using MFCC features)
def extract_features(audio, sample_rate=22050, n_mfcc=13):
    try:
        # Extract 13 MFCC coefficients (mean and std) -> 13 * 2 = 26 features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Combine the MFCC mean and std features
        features = np.hstack([mfccs_mean, mfccs_std])

        # Debug: Ensure exactly 26 features
        assert features.shape[0] == 26, f"Feature extraction mismatch: Expected 26 features, got {features.shape[0]}"

        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to record audio and predict weather condition
def predict_weather_condition(duration=5, sample_rate=22050):
    print(f"Recording for {duration} seconds...")

    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    audio = audio.flatten()  # Convert to 1D array

    # Extract features from the recorded audio
    features = extract_features(audio, sample_rate)
    if features is not None:
        # Standardize the features using the loaded scaler
        features_scaled = scaler.transform([features])
        print(f"Scaled features: {features_scaled}")  # Debug print

        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_label = prediction[0]
        predicted_weather = weather_conditions[predicted_label]  # Map the index to weather condition
        print(f"Predicted Weather Condition: {predicted_weather}")
    else:
        print("Error: Could not extract features from the audio.")

# Run prediction for 5 seconds
predict_weather_condition(duration=5)
