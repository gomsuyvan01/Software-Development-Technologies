from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import joblib
import seaborn as sns
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time


def evaluate_model(model, data):
    # Include the same features as in training
    X = data.drop("target", axis=1)
    y = data["target"]

    # Make predictions
    predictions = model.predict(X)

    # Evaluate performance
    acc = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='weighted')
    cm = confusion_matrix(y, predictions)

    # Save confusion matrix as an image file
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Generate a unique filename using timestamp
    filename = f'confusion_matrix_{int(time.time())}.png'
    plt.savefig(filename)
    plt.close()

    return acc, f1


if __name__ == "__main__":
    # Read cleaned data
    data = pd.read_csv("cleaned_data.csv")

    # Load trained model
    model = joblib.load("iris_model.pkl")

    acc, f1 = evaluate_model(model, data)

    print(f"Accuracy: {acc:.2f}")
    print(f"F1-Score: {f1:.2f}")