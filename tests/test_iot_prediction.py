import joblib

from banana_ml.features.iot_features import prepare_iot_features


MODEL_PATH = "models/development/banana_quality_histgradientboosting.joblib"


def test_iot_prediction_classes():
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]

    test_cases = [
        (
            {
                "temperature": 13.6,
                "humidity": 90.0,
                "mq2_raw": 800,
                "fc28_raw": 3000,
                "liquid_detected": 0,
                "motion_detected": 1,
                "ldr_raw": 2100,
            },
            "bon",
        ),
        (
            {
                "temperature": 15.5,
                "humidity": 80.0,
                "mq2_raw": 1250,
                "fc28_raw": 2400,
                "liquid_detected": 0,
                "motion_detected": 1,
                "ldr_raw": 2100,
            },
            "moyen",
        ),
        (
            {
                "temperature": 20.5,
                "humidity": 68.0,
                "mq2_raw": 1750,
                "fc28_raw": 1500,
                "liquid_detected": 1,
                "motion_detected": 1,
                "ldr_raw": 2100,
            },
            "mauvais",
        ),
    ]

    for data, expected in test_cases:
        features = prepare_iot_features(data)
        prediction = model.predict(features)[0]

        assert prediction == expected