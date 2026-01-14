import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

# Preprocess the data
def preprocess_data(data):
    # Add a new feature: ratio of sepal length to petal length
    data['sepal_petal_ratio'] = data['sepal length (cm)'] / data['petal length (cm)']

    # Standardize features (exclude target and ratio)
    scaler = StandardScaler()
    features = data.drop(columns=['target', 'sepal_petal_ratio'])
    data[features.columns] = scaler.fit_transform(features)

    # Ensure target is integer
    data['target'] = data['target'].astype(int)

    return data

if name == "__main__":
    data = load_data()
    data = preprocess_data(data)

    # Save cleaned data (ignored by Git)
    data.to_csv("cleaned_data.csv", index=False)
    print("Preprocessed data saved as 'cleaned_data.csv'")