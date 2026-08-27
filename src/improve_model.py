import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# 1. LOAD DATASET
# =========================

data = pd.read_csv("data/data.csv", sep=";")


# =========================
# 2. FEATURES AND TARGET
# =========================

X = data.drop("Target", axis=1)
y = data["Target"]


# =========================
# 3. FEATURE TYPES
# =========================

categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
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

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# =========================
# 4. PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# =========================
# 5. TRAIN/TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 6. BALANCED LOGISTIC
#    REGRESSION
# =========================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


# =========================
# 7. CREATE PIPELINE
# =========================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# =========================
# 8. TRAIN
# =========================

print("Training balanced Logistic Regression...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# =========================
# 9. PREDICT
# =========================

y_pred = pipeline.predict(X_test)


# =========================
# 10. EVALUATE
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n===== BALANCED LOGISTIC REGRESSION =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))