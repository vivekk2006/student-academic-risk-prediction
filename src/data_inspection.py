import pandas as pd

# Load the UCI student dataset
data = pd.read_csv("data/data.csv", sep=";")

# Display basic information
print("Dataset loaded successfully!")
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nTarget distribution:")
print(data["Target"].value_counts())