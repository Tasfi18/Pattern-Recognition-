# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv(r"C:\Users\ADMIN\Desktop\pattern recognition\iris.csv")
data = data.drop("Id", axis=1)

# ----------------------------
# KNN Class Implementation
# ----------------------------
class KNN:
    def __init__(self, k):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = []

        for x in X_test:
            # Compute squared Euclidean distances
            distances = [ (np.sum((x - x_train) ** 2), label) 
                          for x_train, label in zip(self.X_train, self.y_train) ]

            # Sort by distance and select k nearest
            distances.sort(key=lambda d: d[0])
            neighbors = distances[:self.k]

            # Count majority label
            labels, counts = np.unique([label for _, label in neighbors], return_counts=True)
            majority_label = labels[np.argmax(counts)]
            y_pred.append(majority_label)

        return y_pred

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Species", axis=1), 
    data["Species"], 
    test_size=0.5, 
    random_state=42
)

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate_knn(k_values, X_train, X_test, y_train, y_test):
    accuracies = []
    for k in k_values:
        model = KNN(k)
        model.train(X_train, y_train)
        prediction = model.predict(X_test)

        acc = accuracy_score(y_test, prediction)
        accuracies.append(acc)

        print(f"\nK = {k}, Accuracy = {acc*100:.2f}%")
        cm = confusion_matrix(y_test, prediction)

        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=np.unique(y_test), 
                    yticklabels=np.unique(y_test))
        plt.title(f"Confusion Matrix (k={k})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    return accuracies

# ----------------------------
# Run for k = 1, 3, 5, 7, 9
# ----------------------------
k_values = [1, 3, 5, 7, 9]
accuracies = evaluate_knn(k_values, X_train, X_test, y_train, y_test)

# ----------------------------
# Plot Accuracy vs K
# ----------------------------
plt.figure(figsize=(6,4))
plt.plot(k_values, accuracies, marker="o")
plt.title("Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
