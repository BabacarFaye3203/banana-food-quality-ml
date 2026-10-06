from fastapi import FastAPI,HTTPException

from banana_ml.api_schemas import IoTMeasurementRequest
from banana_ml.models.predictor import predict_quality


app = FastAPI(
    title="Banana Quality ML API",
    description="API de classification de la qualité des bananes à partir des données IoT.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "banana-quality-ml-api",
    }


@app.post("/predict")
def predict(data: IoTMeasurementRequest):
    try:
        quality = predict_quality({
            "temperature": data.temperature,
            "humidity": data.humidity,
            "mq2_raw": data.mq2_raw,
            "fc28_raw": data.fc28_raw,
            "liquid_detected": data.liquid_detected,
            "motion_detected": data.motion_detected,
            "ldr_raw": data.ldr_raw,
        })

    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "device_id": data.device_id,
        "quality": quality,
        "model_version": "1.0.0",
    }