import pandas as pd

from banana_ml.data.validation import validate_iot_dataset


def test_iot_dataset_is_valid():
    df = pd.read_csv("data/raw/banana_iot_model_ready.csv")

    validate_iot_dataset(df)