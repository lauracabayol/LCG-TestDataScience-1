from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from VitalMetrics.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
@app.command()
def main(
    # Paths for input and output files
    input_path: Path = RAW_DATA_DIR / "palmer-penguins-dataset-for-eda/penguins.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "penguin_train_features.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "penguin_test_features.csv",
    test_size: float = 0.2,  # Size of the test set (20% by default)
    random_state: int = 42,  # Reproducibility
    verbose: bool = True     # Option to print out progress
):
    """Function to load dataset, split into train/test, generate features, and save to output files."""

    # Load dataset
    logger.info(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        logger.success("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Split the dataset into train and test sets
    logger.info(f"Splitting the dataset into train/test with test size {test_size}...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    if verbose:
        logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples.")

    # Feature engineering (encoding and scaling)
    train_df, test_df = feature_engineering(train_df, test_df, verbose)

    # Save the transformed features
    logger.info(f"Saving processed training features to {train_output_path}...")
    try:
        train_df.to_csv(train_output_path, index=False)
        logger.success("Training features saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save training features: {e}")

    logger.info(f"Saving processed test features to {test_output_path}...")
    try:
        test_df.to_csv(test_output_path, index=False)
        logger.success("Test features saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save test features: {e}")


def feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = True) -> (pd.DataFrame, pd.DataFrame):
    """Performs feature engineering on the train and test datasets."""

    if verbose:
        logger.info("Starting feature engineering...")

    # Drop rows with missing values
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # Define columns to delete, encode, and scale
    columns_to_delete = ['sex', 'year']
    columns_to_encode = ['species','island']
    features_to_scale = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']  # Numeric features to scale

    # Delete columns
    train_df.drop(columns = columns_to_delete, inplace=True)
    test_df.drop(columns = columns_to_delete, inplace=True)
    
    # Encoding categorical columns (using LabelEncoder)
    logger.info("Applying label encoding on categorical features...")
    LabEnc_mapping = {}
    for col in columns_to_encode:
        label_encoder = LabelEncoder()
        train_df[col] = label_encoder.fit_transform(train_df[col].values)
        test_df[col] = label_encoder.transform(test_df[col].values)
        LabEnc_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Scaling numerical columns (fit only on the training set)
    scaler = StandardScaler()
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])  # Transform the test set using the same scaler

    if verbose:
        logger.info("Feature engineering completed.")

    return train_df, test_df


if __name__ == "__main__":
    app()
