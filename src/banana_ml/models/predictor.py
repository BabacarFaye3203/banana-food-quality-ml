import mlflow
import pandas as pd

from banana_ml.data.validation import validate_iot_measurement
mlflow.set_tracking_uri("http://host.docker.internal:5000")
MODEL_URI = "models:/banana_quality_model/1"

_model = None


def get_model():
    global _model

    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)

    return _model


def predict_quality(iot_data: dict) -> str:
    validate_iot_measurement(iot_data)

    features = pd.DataFrame(
        [
            {
                "temperature": iot_data["temperature"],
                "humidity": iot_data["humidity"],
                "mq2_raw": iot_data["mq2_raw"],
                "fc28_raw": iot_data["fc28_raw"],
                "liquid_detected": iot_data["liquid_detected"],
                "motion_detected": iot_data["motion_detected"],
                "ldr_raw": iot_data["ldr_raw"],
            }
        ]
    )

    model = get_model()

    prediction = model.predict(features)

    return str(prediction[0])