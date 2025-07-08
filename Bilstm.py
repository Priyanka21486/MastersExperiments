import os
from sklearn.model_selection import train_test_split
from glob import glob
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def preprocess_audio(file_path):
    y, sr = librosa.load(file_path)
    normalized_y = librosa.util.normalize(y)
    mfccs = librosa.feature.mfcc(y=normalized_y, sr=sr)
    return mfccs

def load_data_from_folder(folder_path, class_label):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            mfccs = preprocess_audio(os.path.join(folder_path, filename))
            X.append(mfccs)
            y.append(class_label)
    return np.array(X), np.array(y)

# Assuming folder names are 'class_0' and 'class_1'
folder_paths = ['/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_INT', '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_INT']
class_labels = [0, 1]  # Corresponding labels for each folder

X_all = []
y_all = []

for folder_path, class_label in zip(folder_paths, class_labels):
    X, y = load_data_from_folder(folder_path, class_label)
    X_all.extend(X)
    y_all.extend(y)

# Convert lists to numpy arrays
X_all = np.array(X_all)
y_all = np.array(y_all)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Reshape inputs for LSTM layers
input_shape = (X_train.shape[1], X_train.shape[2])  # Assuming MFCCs are stored as (n_mfccs, time_steps)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

def create_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

model = create_model(input_shape)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
print(y_pred)

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Plot the confusion matrix
