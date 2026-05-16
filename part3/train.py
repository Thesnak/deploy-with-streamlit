import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/breast_cancer_pipeline.pkl"
DATA_PATH = "models/breast_cancer_train_data.pkl"


def train_and_save_model():
    """Train the model and save the pipeline + training metadata."""
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "target_names": list(data.target_names),
        "stats": X_train.describe().T
    }, DATA_PATH)

    print(f"Saved trained model to {MODEL_PATH}")
    print(f"Saved train metadata to {DATA_PATH}")


def load_pipeline_and_data():
    """Load a saved pipeline and training metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing model files. Run 'python train.py' from the part3 directory first."
        )

    pipeline = joblib.load(MODEL_PATH)
    train_data = joblib.load(DATA_PATH)
    return pipeline, train_data


if __name__ == "__main__":
    train_and_save_model()
