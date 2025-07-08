import scipy.io

# Replace 'yourfile.mat' with the path to your .mat file
file_path = '/home/spl_cair/Desktop/priyanka/Priyanka_results_aug_2024.mat'

# Load the .mat file
data = scipy.io.loadmat(file_path)

# Print the keys of the dictionary to see what variables are in the file
# print("Keys in the .mat file:", data.keys())
Healthy_lpcc = data['Healthy_lpcc']
# print("Contents of 'Healthy_STCC':", Healthy_STCC)
# print(type(Healthy_STCC), Healthy_STCC.shape)
Stutter_lpcc = data['Stutter_lpcc']
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Example arrays
class_1 = Healthy_lpcc
class_2 = Stutter_lpcc
print(class_1)
print(f"class2-----------{class_2}")

# Combine the data
X = np.vstack((class_1, class_2))
y = np.hstack((np.zeros(class_1.shape[0]), np.ones(class_2.shape[0])))
print(X.shape)
print(y.shape)
nan_mask = np.isnan(X)

# Count the number of NaNs
num_nans = np.sum(nan_mask)

# print("Number of NaNs in the array:", num_nans)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train the classifier
clf = SVC(kernel='poly')



# from sklearn.impute import SimpleImputer

# # Create an imputer object with a strategy to remove missing values
# imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or 'most_frequent'

# # Fit the imputer on the training data and transform both train and test data
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)

# Train the classifier with imputed data
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
# print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)