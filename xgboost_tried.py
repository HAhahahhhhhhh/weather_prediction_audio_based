import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib  # For saving and loading the model

# Step 1: Load and Preprocess Dataset
file_path = 'data/audio_features.csv'  # Path to your dataset
data = pd.read_csv(file_path)

# Separate features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data['label']  # Target column

# Encode target labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute class weights to handle potential imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))

# Convert class_weight_dict to an array that will be used for sample_weight
sample_weights = np.array([class_weight_dict[class_label] for class_label in y_encoded])

# Step 2: Train the XGBoost Model using Stratified Cross-Validation
model = xgb.XGBClassifier(n_estimators=200, random_state=42, learning_rate=0.01, max_depth=6, subsample=0.8)

# Stratified K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for train_idx, test_idx in kfold.split(X_scaled, y_encoded):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Generate sample weights for the current split
    train_sample_weights = sample_weights[train_idx]

    # Fit model with class weights (sample_weight)
    model.fit(X_train, y_train, sample_weight=train_sample_weights)

    # Evaluate model
    y_pred = model.predict(X_test)
    cv_results.append(accuracy_score(y_test, y_pred))

print(f"Cross-validation accuracy: {np.mean(cv_results):.4f}")

# Step 3: Train on the Entire Dataset and Compute Overall Metrics
model.fit(X_scaled, y_encoded, sample_weight=sample_weights)

# Predict on the entire dataset for confusion matrix and accuracy
y_pred_overall = model.predict(X_scaled)

# Compute accuracy
overall_accuracy = accuracy_score(y_encoded, y_pred_overall)
print(f"\nOverall Model Accuracy: {overall_accuracy:.4f}")

# Compute confusion matrix
cm_overall = confusion_matrix(y_encoded, y_pred_overall)
print("\nOverall Confusion Matrix:")
print(cm_overall)

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_encoded, y_pred_overall, target_names=label_encoder.classes_))

# Step 4: Save the trained model to a file
def save_model(model, filename='weather_predictor_model.json'):
    """
    Save the trained XGBoost model to a file.
    """
    model.save_model(filename)  # Save as JSON (XGBoost's built-in method)

# Save the trained model
save_model(model, 'model_save/weather_predictor_model.json')

# Optionally, save the scaler (if you want to use it for future feature scaling)
joblib.dump(scaler, 'model_save/scaler.pkl')  # Save the scaler for future use
