from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banana_iot_model_ready.csv"
)


FEATURES = [
    "temperature",
    "humidity",
    "mq2_raw",
    "fc28_raw",
    "liquid_detected",
    "motion_detected",
    "ldr_raw",
]

TARGET = "quality"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = HistGradientBoostingClassifier(
        random_state=42,
    )

    mlflow.set_experiment("banana-quality-classification")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "HistGradientBoostingClassifier")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            model,
            name="banana_quality_model",
            serialization_format="cloudpickle",
            registered_model_name="banana_quality_model",
        )

        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()