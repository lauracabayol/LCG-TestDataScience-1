from pathlib import Path
import pandas as pd
from loguru import logger

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering on the train and test datasets."""

    logger.info("Starting feature engineering...")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Define columns to delete, encode, and scale
    columns_to_delete = ["sex", "year"]
    columns_to_encode = ["species", "island"]
    features_to_scale = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    # Delete columns
    df.drop(columns=columns_to_delete, inplace=True)

    # Encoding categorical columns (using LabelEncoder)
    logger.info("Applying label encoding on categorical features...")
    LabEnc_mapping = {}
    for col in columns_to_encode:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].values)
        LabEnc_mapping[col] = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )

    # Scaling numerical columns
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    logger.info("Feature engineering completed.")

    return df
