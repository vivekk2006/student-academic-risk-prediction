import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1. LOAD DATASET
# =========================

data = pd.read_csv("data/data.csv", sep=";")


# =========================
# 2. SEPARATE FEATURES
#    AND TARGET
# =========================

X = data.drop("Target", axis=1)
y = data["Target"]


# =========================
# 3. DEFINE FEATURE TYPES
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
# 6. DEFINE MODELS
# =========================

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


# =========================
# 7. TRAIN MODELS
# =========================

trained_models = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)

    trained_models[model_name] = pipeline

    print(f"{model_name} training completed.")


print("\n===== ALL MODELS TRAINED SUCCESSFULLY =====")