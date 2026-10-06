import pandas as pd

MODEL_FEATURES = [
    "temperature",
    "humidity",
    "mq2_raw",
    "fc28_raw",
    "liquid_detected",
    "motion_detected",
    "ldr_raw",
]


def prepare_iot_features(data: dict) -> pd.DataFrame:
    """
    Transforme une mesure IoT brute en DataFrame compatible
    avec le modèle de classification.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Features IoT manquantes : {missing_features}"
        )

    features = {
        feature: data[feature]
        for feature in MODEL_FEATURES
    }

    return pd.DataFrame([features], columns=MODEL_FEATURES)