import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_dataset_path():
    if os.path.exists("dataset"):
        return "dataset"
    if os.path.exists("datasets/audio"):
        return "datasets/audio"
    if os.path.exists("datasets"):
        return "datasets"
    return "dataset"

DATASET_PATH = get_dataset_path()
MODEL_PATH = "model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Details: {e}")
        return None

def process_directory(dir_path):
    features = []
    labels = []
    for class_name in os.listdir(dir_path):
        class_dir = os.path.join(dir_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.mp3', '.wav', '.ogg', '.flac', '.m4a')):
                file_path = os.path.join(class_dir, file_name)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(class_name)
    return features, labels

def main():
    print(f"Extracting features from dataset directory '{DATASET_PATH}'...")
    features, labels = process_directory(DATASET_PATH)

    if not features:
        print("No audio files processed. Please check your dataset directory.")
        return

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Training on {len(X)} samples across {len(le.classes_)} classes.")
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save model and label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print(f"Created {DATASET_PATH} directory. Please add your audio files in class subfolders.")
    else:
        main()
