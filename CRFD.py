import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
import itertools

data_0 = pd.read_excel('C:/Users/cxq/Desktop/huo8/0/1/1.xlsx', header=None).values.flatten()
data_1 = pd.read_excel('C:/Users/cxq/Desktop/huo8/1/1/1.xlsx', header=None).values.flatten()

def create_slices(data, slice_size=500, step_size=10):
    slices = []
    for start in range(0, len(data) - slice_size + 1, step_size):
        slices.append(data[start:start + slice_size])
    return slices

slices_0 = create_slices(data_0)
slices_1 = create_slices(data_1)

def extract_features(slices):
    features = []
    for s in slices:
        mean_val = np.mean(s)
        var_val = np.var(s)
        instantaneous_rate = np.mean(np.abs(np.diff(s)))
        f, psd = welch(s)
        power_spectral_density = np.mean(psd[f < np.percentile(f, 10)])
        features.append([mean_val, var_val, instantaneous_rate, power_spectral_density])
    return np.array(features)

features_0 = extract_features(slices_0)
features_1 = extract_features(slices_1)

labels_0 = np.zeros(features_0.shape[0])
labels_1 = np.ones(features_1.shape[0])

X = np.vstack((features_0, features_1))
y = np.concatenate((labels_0, labels_1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Log Loss: {loss:.4f}')

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.tight_layout()
plt.show()
