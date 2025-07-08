import torch
import numpy as np
import librosa
import os
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
torch.cuda.empty_cache()
# Define the TDNN model
class TDNNModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(TDNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(256 * 15, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load pre-trained wav2vec 2.0 model and feature extractor
model_name = "facebook/wav2vec2-large-xlsr-53"
model = Wav2Vec2Model.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
device = torch.device("cpu")
model.to('cpu')

def extract_features_from_layer(audio_path, model, layer_index):
    """
    Extract features from a specific intermediate layer of the wav2vec 2.0 model.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, return_tensors="pt", padding="longest", sr=16000)
    inputs = {key: value.to('cpu') for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states  # Tuple of (layer_1, layer_2, ..., layer_n)
    
    if layer_index < 0 or layer_index >= len(hidden_states):
        raise ValueError("Invalid layer index")
    
    layer_features = hidden_states[layer_index]  # [batch_size, sequence_length, hidden_size]
    features = layer_features.cpu().numpy()  # Mean pooling to [batch_size, hidden_size]
    
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
    
    layer_index = 5  # Example layer index (0-based)

    # Extract features from all folders
    X, y = extract_features_from_folders(folder_paths, model, layer_index)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

    # Apply SMOTE to handle class imbalance
    # smote = SMOTE(random_state=42)
    
    # Reshape data for SMOTE
    # Assuming X has shape (num_samples, num_features)
    # X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten to 2D
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

    # Reshape back for TDNN
    # Assuming the original shape before flattening was (num_samples, num_frames, num_features)
    # num_features = X_train.shape[1]  # Number of features
    # num_frames = X_train.shape[2]    # Number of frames
    # X_train_resampled = X_train_resampled.reshape(-1, num_features, num_frames)  # Reshape back to 3D

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Initialize and train the TDNN model
    input_size = X_train_tensor.shape[1]  # Number of features
    num_classes = len(np.unique(y_train))
    tdnn_model = TDNNModel(input_size=input_size, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tdnn_model.parameters(), lr=0.001)

    # Training loop (simplified)
    tdnn_model.train()
    for epoch in range(5):  # Number of epochs
        optimizer.zero_grad()
        outputs = tdnn_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    torch.cuda.empty_cache()

    # Evaluation
    tdnn_model.eval()
    with torch.no_grad():
        outputs = tdnn_model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
    
    accuracy = accuracy_score(y_test_tensor.cpu(), y_pred.cpu())
    print(f"f1: {f1_score(y_test_tensor.cpu(), y_pred.cpu(), average='weighted'):.4f}")
    print(f"auc: {roc_auc_score(y_test_tensor.cpu(), y_pred.cpu(), multi_class='ovr'):.4f}")
    print(f"conf: \n{confusion_matrix(y_test_tensor.cpu(), y_pred.cpu())}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
