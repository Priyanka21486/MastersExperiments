import os
import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

def extract_mfcc_features(audio_path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def load_data_from_folders(folder1, folder2):
    """Load MFCC features and labels from two folders."""
    features = []
    labels = []
    
    # Process folder1
    for filename in os.listdir(folder1):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder1, filename)
            mfccs = extract_mfcc_features(file_path)
            features.append(mfccs)
            labels.append(0)  # Label for folder1
    
    # Process folder2
    for filename in os.listdir(folder2):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder2, filename)
            mfccs = extract_mfcc_features(file_path)
            print(mfccs)
            features.append(mfccs)
            print(np.array(features).shape)
            labels.append(1)  # Label for folder2
    
    return np.array(features), np.array(labels)

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train an SVM classifier and evaluate its performance."""
    # Create a pipeline with standard scaling and SVM classifier
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')



# Example usage
folder_class1 = '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_INT'
folder_class2 = '/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_INT'

# Load data
features, labels = load_data_from_folders(folder_class1, folder_class2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)

# Train and evaluate the model
train_and_evaluate_model(X_train, y_train, X_test, y_test)

