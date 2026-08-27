import pandas as pd


# Baseline model results
results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.7672,
        0.6949,
        0.7537
    ],
    "Precision": [
        0.7541,
        0.7010,
        0.7287
    ],
    "Recall": [
        0.7672,
        0.6949,
        0.7537
    ],
    "F1-Score": [
        0.7567,
        0.6977,
        0.7303
    ]
}


# Create results table
results_df = pd.DataFrame(results)


# Display results
print("===== BASELINE MODEL RESULTS =====")
print(results_df.to_string(index=False))


# Identify the model with the highest F1-score
best_model = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Model"
]

best_f1 = results_df["F1-Score"].max()

print("\n===== BEST BASELINE MODEL =====")
print("Model:", best_model)
print(f"F1-Score: {best_f1:.4f}")