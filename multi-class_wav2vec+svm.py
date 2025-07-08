import torch
import numpy as np
import librosa
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load pre-trained wav2vec 2.0 model and feature extractor
model_name = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2Model.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_features_from_layer(audio_path, model, layer_index):
    """
    Extract features from a specific intermediate layer of the wav2vec 2.0 model.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, return_tensors="pt", padding="longest")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # Tuple of (layer_1, layer_2, ..., layer_n)
    
    if layer_index < 0 or layer_index >= len(hidden_states):
        raise ValueError("Invalid layer index")
    
    layer_features = hidden_states[layer_index]  # [batch_size, sequence_length, hidden_size]
    features = layer_features.mean(dim=1).cpu().numpy()  # Mean pooling to [batch_size, hidden_size]
    
    return features

def extract_features_from_folders(folder_paths, model, layer_index):
    """
    Extract features from all audio files in multiple folders and return the features and labels.
    """
    features = []
    labels = []
    
    for label, folder_path in enumerate(folder_paths):
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(folder_path, filename)
                file_features = extract_features_from_layer(audio_path, model, layer_index)
                features.append(file_features)
                labels.extend([label] * len(file_features))
    
    return np.vstack(features), np.array(labels)

def main():
    # Define folder paths for each class
    folder_paths = [
        "/home/spl_cair/Desktop/priyanka/icassp_exp/IED_REP_OUT",
        "/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_REP_OUT",
        "/home/spl_cair/Desktop/priyanka/icassp_exp/IED_PR",
        "/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_PR",
        "/home/spl_cair/Desktop/priyanka/icassp_exp/IED_INT",
        "/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_INT"
    ]
    
    layer_index = 5 # Example layer index (0-based)

    # Extract features from all folders
    X, y = extract_features_from_folders(folder_paths, model, layer_index)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train and evaluate SVM classifier
    classifier = SVC(kernel='rbf', decision_function_shape='ovr')  # Using RBF kernel and one-vs-rest
    classifier.fit(X_train_resampled, y_train_resampled)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"f1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    # print(f"auc: {roc_auc_score(y_test, classifier.decision_function(X_test), multi_class='ovr'):.4f}")
    print(f"conf: \n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
