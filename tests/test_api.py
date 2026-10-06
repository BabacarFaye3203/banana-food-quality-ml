from fastapi.testclient import TestClient

from banana_ml.api import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "banana-quality-ml-api",
    }

def test_predict_valid_iot_measurement():
    payload = {
        "id": 101,
        "device_id": "esp32-001",
        "temperature": 13.6,
        "humidity": 90.0,
        "mq2_raw": 800,
        "mq2_voltage": 0.64,
        "fc28_raw": 3000,
        "liquid_detected": 0,
        "motion_detected": 1,
        "ldr_raw": 2100,
        "light_label": "bright",
        "detected_at": "2026-10-06T02:30:00",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    assert response.json() == {
        "device_id": "esp32-001",
        "quality": "bon",
        "model_version": "1.0.0",
    }

def test_predict_rejects_invalid_mq2():
    payload = {
        "id": 102,
        "device_id": "esp32-001",
        "temperature": 13.6,
        "humidity": 90.0,
        "mq2_raw": 5000,
        "mq2_voltage": 0.64,
        "fc28_raw": 3000,
        "liquid_detected": 0,
        "motion_detected": 1,
        "ldr_raw": 2100,
        "light_label": "bright",
        "detected_at": "2026-10-06T02:30:00",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "mq2_raw"]

def test_predict_quality_classes():
    scenarios = [
        (
            {
                "temperature": 13.6,
                "humidity": 90.0,
                "mq2_raw": 800,
                "mq2_voltage": 0.64,
                "fc28_raw": 3000,
                "liquid_detected": 0,
                "motion_detected": 1,
                "ldr_raw": 2100,
                "light_label": "bright",
                "detected_at": "2026-10-06T02:30:00",
            },
            "bon",
        ),
        (
            {
                "temperature": 15.5,
                "humidity": 80.0,
                "mq2_raw": 1250,
                "mq2_voltage": 1.01,
                "fc28_raw": 2400,
                "liquid_detected": 0,
                "motion_detected": 1,
                "ldr_raw": 2100,
                "light_label": "bright",
                "detected_at": "2026-10-06T02:30:00",
            },
            "moyen",
        ),
        (
            {
                "temperature": 20.5,
                "humidity": 68.0,
                "mq2_raw": 1750,
                "mq2_voltage": 1.40,
                "fc28_raw": 1900,
                "liquid_detected": 1,
                "motion_detected": 1,
                "ldr_raw": 2100,
                "light_label": "bright",
                "detected_at": "2026-10-06T02:30:00",
            },
            "mauvais",
        ),
    ]

    for payload, expected_quality in scenarios:
        payload["device_id"] = "esp32-001"

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json()["quality"] == expected_quality