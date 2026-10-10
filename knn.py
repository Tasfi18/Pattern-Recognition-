# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_iris():
    data = pd.read_csv(HERE / "iris.csv")
    return data.drop("Id", axis=1)

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
        labels = np.unique(np.concatenate([np.array(y_test), np.array(prediction)]))
        cm = confusion_matrix(y_test, prediction, labels=labels)

        # Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=labels, 
                    yticklabels=labels)
        plt.title(f"Confusion Matrix (k={k})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    return accuracies

if __name__ == "__main__":
    data = load_iris()

    # Half the flowers for training, half for testing
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Species", axis=1),
        data["Species"],
        test_size=0.5,
        random_state=42
    )

    # Run for k = 1, 3, 5, 7, 9
    k_values = [1, 3, 5, 7, 9]
    accuracies = evaluate_knn(k_values, X_train, X_test, y_train, y_test)

    # Plot accuracy vs k
    plt.figure(figsize=(6,4))
    plt.plot(k_values, accuracies, marker="o")
    plt.title("Accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

