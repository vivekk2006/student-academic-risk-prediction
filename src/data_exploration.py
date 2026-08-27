import pandas as pd

# Load the dataset
data = pd.read_csv("data/data.csv", sep=";")

print("===== DATA TYPES =====")
print(data.dtypes)

print("\n===== DUPLICATE ROWS =====")
print("Number of duplicate rows:", data.duplicated().sum())

print("\n===== TARGET COUNTS =====")
print(data["Target"].value_counts())

print("\n===== TARGET PERCENTAGES =====")
print((data["Target"].value_counts(normalize=True) * 100).round(2))

print("\n===== NUMERICAL SUMMARY =====")
print(data.describe().T)