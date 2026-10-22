from banana_ml.models.predictor import predict_quality
import pytest



def test_predict_quality_bon():
    data = {
        "temperature": 13.6,
        "humidity": 90.0,
        "mq2_raw": 800,
        "fc28_raw": 3000,
        "liquid_detected": 0,
        "motion_detected": 1,
        "ldr_raw": 2100,
    }

    assert predict_quality(data) == "bon"


def test_predict_quality_moyen():
    data = {
        "temperature": 15.5,
        "humidity": 80.0,
        "mq2_raw": 1250,
        "fc28_raw": 2400,
        "liquid_detected": 0,
        "motion_detected": 1,
        "ldr_raw": 2100,
    }

    assert predict_quality(data) == "moyen"


def test_predict_quality_mauvais():
    data = {
        "temperature": 20.5,
        "humidity": 68.0,
        "mq2_raw": 1750,
        "fc28_raw": 1500,
        "liquid_detected": 1,
        "motion_detected": 1,
        "ldr_raw": 2100,
    }

    assert predict_quality(data) == "mauvais"


def test_predict_quality_rejects_invalid_iot_data():
    data = {
        "temperature": 13.6,
        "humidity": 90.0,
        "mq2_raw": 5000,
        "fc28_raw": 3000,
        "liquid_detected": 0,
        "motion_detected": 1,
        "ldr_raw": 2100,
    }

    with pytest.raises(ValueError, match="mq2_raw"):
        predict_quality(data)