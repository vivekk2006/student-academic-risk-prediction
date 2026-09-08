import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/data.csv"
RESULTS_PATH = "week 3/results"

os.makedirs(RESULTS_PATH, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv(DATA_PATH, sep=";")

print("=" * 70)
print("WEEK 3 - EXPERIMENTAL DESIGN AND MODEL EVALUATION")
print("=" * 70)

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

data.columns = data.columns.str.strip()

print("\nColumn names cleaned successfully.")


# ============================================================
# 4. DEFINE TARGET
# ============================================================

target_column = "Target"

X = data.drop(columns=[target_column])
y = data[target_column]


# ============================================================
# 5. REMOVE SECOND-SEMESTER FEATURES
# ============================================================

# The project focuses on early prediction at the end of
# the first semester. Therefore, second-semester information
# is removed to prevent data leakage.

second_semester_features = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

existing_second_semester_features = [
    col for col in second_semester_features
    if col in X.columns
]

X = X.drop(columns=existing_second_semester_features)


print("\nSecond-semester features removed:")

for feature in existing_second_semester_features:
    print(" -", feature)

print("\nNumber of features used for early prediction:", X.shape[1])


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 7. DEFINE CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

categorical_features = [
    col for col in categorical_features
    if col in X.columns
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]


print("\nCategorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split:")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ============================================================
# 10. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 11. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted"
}

cv_results = []

print("\n")
print("=" * 70)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)


for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

    cv_results.append({
        "Model": model_name,
        "CV Accuracy": results["test_accuracy"].mean(),
        "CV Precision Macro": results["test_precision_macro"].mean(),
        "CV Recall Macro": results["test_recall_macro"].mean(),
        "CV F1 Macro": results["test_f1_macro"].mean(),
        "CV F1 Weighted": results["test_f1_weighted"].mean()
    })

    print("\n" + model_name)
    print("-" * 50)

    print(
        f"Accuracy       : "
        f"{results['test_accuracy'].mean():.4f}"
    )

    print(
        f"Precision Macro: "
        f"{results['test_precision_macro'].mean():.4f}"
    )

    print(
        f"Recall Macro   : "
        f"{results['test_recall_macro'].mean():.4f}"
    )

    print(
        f"F1 Macro       : "
        f"{results['test_f1_macro'].mean():.4f}"
    )

    print(
        f"F1 Weighted    : "
        f"{results['test_f1_weighted'].mean():.4f}"
    )


# ============================================================
# 12. CROSS-VALIDATION SUMMARY
# ============================================================

cv_summary = pd.DataFrame(cv_results)

print("\n")
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    cv_summary.round(4).to_string(index=False)
)

cv_summary.to_csv(
    os.path.join(
        RESULTS_PATH,
        "cross_validation_results.csv"
    ),
    index=False
)


# ============================================================
# 13. HELD-OUT TEST SET EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("HELD-OUT TEST SET RESULTS")
print("=" * 70)

test_results = []

test_predictions = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    test_predictions[model_name] = y_pred

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    test_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision Macro": precision,
        "Recall Macro": recall,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted
    })

    print("\n" + model_name)
    print("-" * 50)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision Macro: {precision:.4f}")
    print(f"Recall Macro   : {recall:.4f}")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Weighted    : {f1_weighted:.4f}")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    # Save classification report
    safe_model_name = model_name.lower().replace(" ", "_")

    with open(
        os.path.join(
            RESULTS_PATH,
            f"classification_report_{safe_model_name}.txt"
        ),
        "w"
    ) as file:

        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "Dropout",
            "Enrolled",
            "Graduate"
        ]
    )

    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix as CSV
    cm_dataframe = pd.DataFrame(
        cm,
        index=[
            "Actual Dropout",
            "Actual Enrolled",
            "Actual Graduate"
        ],
        columns=[
            "Predicted Dropout",
            "Predicted Enrolled",
            "Predicted Graduate"
        ]
    )

    cm_dataframe.to_csv(
        os.path.join(
            RESULTS_PATH,
            f"confusion_matrix_{safe_model_name}.csv"
        )
    )

    # Create confusion matrix figure
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Dropout",
            "Enrolled",
            "Graduate"
        ],
        yticklabels=[
            "Dropout",
            "Enrolled",
            "Graduate"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            f"confusion_matrix_{safe_model_name}.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 14. TEST SET SUMMARY
# ============================================================

test_summary = pd.DataFrame(test_results)

print("\n")
print("=" * 70)
print("TEST SET SUMMARY")
print("=" * 70)

print(
    test_summary.round(4).to_string(index=False)
)

test_summary.to_csv(
    os.path.join(
        RESULTS_PATH,
        "model_results.csv"
    ),
    index=False
)


# ============================================================
# 15. MULTICLASS ROC-AUC
# ============================================================

print("\n")
print("=" * 70)
print("MULTICLASS ROC-AUC")
print("=" * 70)

roc_auc_results = []

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    if hasattr(pipeline, "predict_proba"):

        y_probability = pipeline.predict_proba(X_test)

        roc_auc = roc_auc_score(
            y_test,
            y_probability,
            multi_class="ovr",
            average="macro"
        )

        roc_auc_results.append({
            "Model": model_name,
            "ROC-AUC OVR Macro": roc_auc
        })

        print(
            f"{model_name}: {roc_auc:.4f}"
        )


roc_auc_summary = pd.DataFrame(
    roc_auc_results
)

roc_auc_summary.to_csv(
    os.path.join(
        RESULTS_PATH,
        "roc_auc_results.csv"
    ),
    index=False
)


# ============================================================
# 16. CLASS IMBALANCE EXPERIMENT
# ============================================================

print("\n")
print("=" * 70)
print("CLASS-BALANCED LOGISTIC REGRESSION EXPERIMENT")
print("=" * 70)

balanced_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

balanced_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", balanced_model)
    ]
)

balanced_pipeline.fit(
    X_train,
    y_train
)

balanced_pred = balanced_pipeline.predict(
    X_test
)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_pred
)

balanced_precision = precision_score(
    y_test,
    balanced_pred,
    average="macro",
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_pred,
    average="macro",
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_pred,
    average="macro",
    zero_division=0
)

print(
    f"\nAccuracy       : {balanced_accuracy:.4f}"
)

print(
    f"Precision Macro: {balanced_precision:.4f}"
)

print(
    f"Recall Macro   : {balanced_recall:.4f}"
)

print(
    f"F1 Macro       : {balanced_f1:.4f}"
)

print("\nClassification Report:")

balanced_report = classification_report(
    y_test,
    balanced_pred,
    zero_division=0
)

print(balanced_report)

with open(
    os.path.join(
        RESULTS_PATH,
        "classification_report_balanced_logistic_regression.txt"
    ),
    "w"
) as file:

    file.write(balanced_report)


balanced_cm = confusion_matrix(
    y_test,
    balanced_pred,
    labels=[
        "Dropout",
        "Enrolled",
        "Graduate"
    ]
)

print("Confusion Matrix:")
print(balanced_cm)

balanced_cm_dataframe = pd.DataFrame(
    balanced_cm,
    index=[
        "Actual Dropout",
        "Actual Enrolled",
        "Actual Graduate"
    ],
    columns=[
        "Predicted Dropout",
        "Predicted Enrolled",
        "Predicted Graduate"
    ]
)

balanced_cm_dataframe.to_csv(
    os.path.join(
        RESULTS_PATH,
        "confusion_matrix_balanced_logistic_regression.csv"
    )
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    balanced_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Dropout",
        "Enrolled",
        "Graduate"
    ],
    yticklabels=[
        "Dropout",
        "Enrolled",
        "Graduate"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    "Confusion Matrix - Balanced Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "confusion_matrix_balanced_logistic_regression.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. COMPARE STANDARD AND BALANCED LOGISTIC REGRESSION
# ============================================================

standard_lr = test_summary[
    test_summary["Model"] == "Logistic Regression"
].iloc[0]

balanced_comparison = pd.DataFrame([
    {
        "Model": "Standard Logistic Regression",
        "Accuracy": standard_lr["Accuracy"],
        "Precision Macro": standard_lr["Precision Macro"],
        "Recall Macro": standard_lr["Recall Macro"],
        "F1 Macro": standard_lr["F1 Macro"]
    },
    {
        "Model": "Balanced Logistic Regression",
        "Accuracy": balanced_accuracy,
        "Precision Macro": balanced_precision,
        "Recall Macro": balanced_recall,
        "F1 Macro": balanced_f1
    }
])

print("\n")
print("=" * 70)
print("STANDARD VS BALANCED LOGISTIC REGRESSION")
print("=" * 70)

print(
    balanced_comparison.round(4).to_string(index=False)
)

balanced_comparison.to_csv(
    os.path.join(
        RESULTS_PATH,
        "balanced_logistic_regression_comparison.csv"
    ),
    index=False
)


# ============================================================
# 18. ERROR ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("ERROR ANALYSIS - LOGISTIC REGRESSION")
print("=" * 70)

lr_predictions = test_predictions[
    "Logistic Regression"
]

error_analysis = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": lr_predictions
})

errors = error_analysis[
    error_analysis["Actual"] != error_analysis["Predicted"]
]

print("\nTotal test samples:", len(error_analysis))
print("Total incorrect predictions:", len(errors))
print(
    "Error rate:",
    round(
        len(errors) / len(error_analysis) * 100,
        2
    ),
    "%"
)

print("\nError distribution:")
print(
    errors.groupby(
        ["Actual", "Predicted"]
    ).size()
)

# Save error analysis
errors.to_csv(
    os.path.join(
        RESULTS_PATH,
        "logistic_regression_errors.csv"
    ),
    index=False
)

error_counts = (
    errors.groupby(
        ["Actual", "Predicted"]
    )
    .size()
    .reset_index(name="Count")
)

error_counts.to_csv(
    os.path.join(
        RESULTS_PATH,
        "logistic_regression_error_summary.csv"
    ),
    index=False
)


# ============================================================
# 19. ENROLLED CLASS ERROR ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("ENROLLED CLASS ERROR ANALYSIS")
print("=" * 70)

enrolled_actual = error_analysis[
    error_analysis["Actual"] == "Enrolled"
]

print(
    "\nActual Enrolled students:",
    len(enrolled_actual)
)

print("\nPredictions for actual Enrolled students:")

print(
    enrolled_actual["Predicted"].value_counts()
)

enrolled_error_summary = (
    enrolled_actual["Predicted"]
    .value_counts()
    .reset_index()
)

enrolled_error_summary.columns = [
    "Predicted Class",
    "Count"
]

enrolled_error_summary.to_csv(
    os.path.join(
        RESULTS_PATH,
        "enrolled_class_error_analysis.csv"
    ),
    index=False
)


# ============================================================
# 20. SAVE EXPERIMENT INFORMATION
# ============================================================

experiment_information = {
    "Dataset Rows": data.shape[0],
    "Original Features": data.shape[1] - 1,
    "Features Used": X.shape[1],
    "Removed Second Semester Features": len(
        existing_second_semester_features
    ),
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test),
    "Cross Validation": "5-Fold Stratified",
    "Test Split": "80% Training / 20% Testing",
    "Random State": 42,
    "Target Classes": "Dropout, Enrolled, Graduate"
}

experiment_info_df = pd.DataFrame(
    list(experiment_information.items()),
    columns=["Parameter", "Value"]
)

experiment_info_df.to_csv(
    os.path.join(
        RESULTS_PATH,
        "experiment_information.csv"
    ),
    index=False
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("WEEK 3 EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
The experiment included:

1. End-of-first-semester feature selection
2. Second-semester leakage prevention
3. Stratified 80/20 train-test split
4. 5-fold stratified cross-validation
5. Logistic Regression
6. Decision Tree
7. Random Forest
8. Accuracy evaluation
9. Precision evaluation
10. Recall evaluation
11. Macro F1 evaluation
12. Weighted F1 evaluation
13. Per-class classification analysis
14. Confusion matrix analysis
15. Multiclass ROC-AUC
16. Class imbalance experiment
17. Logistic Regression error analysis
18. Enrolled-class error analysis
19. Results saved to week 3/results
""")

print("\nSaved result files to:")
print(RESULTS_PATH)