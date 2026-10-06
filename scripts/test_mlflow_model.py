import mlflow
import pandas as pd

MODEL_URI = "models:/banana_quality_model/1"

model = mlflow.pyfunc.load_model(MODEL_URI)

data = pd.DataFrame(
    [
        {
            "temperature": 13.6,
            "humidity": 90.0,
            "mq2_raw": 800,
            "fc28_raw": 3000,
            "liquid_detected": 0,
            "motion_detected": 1,
            "ldr_raw": 2100,
        }
    ]
)

prediction = model.predict(data)

print("Prediction :", prediction[0])