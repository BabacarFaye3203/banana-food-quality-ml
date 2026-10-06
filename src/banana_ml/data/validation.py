from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "ethylene",
    "O2",
    "CO2",
    "days",
    "brix",
    "pH",
    "firmness",
    "color_index",
]

IOT_FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "mq2_raw",
    "mq2_voltage",
    "fc28_raw",
    "liquid_detected",
    "motion_detected",
    "ldr_raw",
    "hour",
    "day_of_week",
    "storage_day",
    "light_bright",
]

TARGET_COLUMN = "quality"

ALLOWED_QUALITY = {
    "bon",
    "moyen",
    "mauvais",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the dataset structure."""

    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    """Ensure that the dataset contains no missing values."""

    missing = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()

    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Missing values detected:\n{missing}"
        )


def validate_duplicates(df: pd.DataFrame) -> None:
    """Ensure that the dataset contains no duplicated rows."""

    duplicates = df.duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Dataset contains {duplicates} duplicated rows."
        )


def validate_numeric_features(df: pd.DataFrame) -> None:
    """Ensure that all model features are numeric."""

    non_numeric = [
        column
        for column in FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric features detected: {non_numeric}"
        )


def validate_target(df: pd.DataFrame) -> None:
    """Validate target values."""

    invalid_values = set(df[TARGET_COLUMN].unique()) - ALLOWED_QUALITY

    if invalid_values:
        raise ValueError(
            f"Invalid quality values: {sorted(invalid_values)}"
        )


def validate_dataset(df: pd.DataFrame) -> None:
    """Run all dataset validation checks."""

    validate_schema(df)
    validate_missing_values(df)
    validate_duplicates(df)
    validate_numeric_features(df)
    validate_target(df)

def validate_iot_schema(df: pd.DataFrame) -> None:
    """Validate the IoT dataset structure."""

    expected_columns = set(IOT_FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise ValueError(
            f"Missing IoT columns: {sorted(missing_columns)}"
        )


def validate_iot_numeric_features(df: pd.DataFrame) -> None:
    """Ensure that IoT model features are numeric."""

    non_numeric = [
        column
        for column in IOT_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric IoT features detected: {non_numeric}"
        )
    
def validate_iot_missing_values(df: pd.DataFrame) -> None:
    """Check missing values in the IoT dataset."""
    missing = df[IOT_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            f"Missing values detected in IoT dataset:\n{missing}"
        )

def validate_iot_dataset(df: pd.DataFrame) -> None:
    """Run IoT dataset validation checks."""
    validate_iot_schema(df)
    validate_iot_missing_values(df)
    validate_duplicates(df)
    validate_iot_numeric_features(df)
    validate_target(df)

def validate_iot_measurement(data: dict) -> None:
    """
    Valide une mesure IoT individuelle avant la prédiction.
    Lève une exception si les données sont invalides.
    """

    required_fields = [
        "temperature",
        "humidity",
        "mq2_raw",
        "fc28_raw",
        "liquid_detected",
        "motion_detected",
        "ldr_raw",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            f"Champs IoT manquants : {missing_fields}"
        )

    for field in required_fields:
        if not isinstance(data[field], (int, float)):
            raise TypeError(
                f"La valeur de '{field}' doit être numérique."
            )

    binary_fields = [
        "liquid_detected",
        "motion_detected",
    ]

    for field in binary_fields:
        if data[field] not in (0, 1):
            raise ValueError(
                f"La valeur de '{field}' doit être 0 ou 1."
            )

    if data["mq2_raw"] < 0 or data["mq2_raw"] > 4095:
        raise ValueError(
            "mq2_raw doit être compris entre 0 et 4095."
        )

    if data["fc28_raw"] < 0 or data["fc28_raw"] > 4095:
        raise ValueError(
            "fc28_raw doit être compris entre 0 et 4095."
        )

    if data["ldr_raw"] < 0 or data["ldr_raw"] > 4095:
        raise ValueError(
            "ldr_raw doit être compris entre 0 et 4095."
        )