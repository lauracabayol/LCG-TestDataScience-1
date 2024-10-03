from tqdm import tqdm
from loguru import logger
from pathlib import Path
import typer
import pandas as pd

import mlflow
import mlflow.sklearn

from VitalMetrics.config import RAW_DATA_DIR, PREDICTED_DATA_DIR
from VitalMetrics.features import feature_engineering

mlflow.set_tracking_uri("http://127.0.0.1:5000")
app = typer.Typer()


@app.command()
def main(
    # ---- Define input and output paths ----
    test_data_path: Path = RAW_DATA_DIR / "palmer-penguins-dataset-for-eda/penguins_test.csv",
    model_name: str = "PENGUINS_CLASSIFIER",
    model_version: int = 16,
    predictions_path: Path = PREDICTED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # Load the model from MLflow
    logger.info(f"Loading model '{model_name}' version {model_version}...")
    model_uri = f"models:/{model_name}/{model_version}"

    # Start MLflow run for logging (optional)
    mlflow.start_run()

    # Load the features for prediction
    logger.info(f"Loading features from {test_data_path}...")
    try:
        df = pd.read_csv(test_data_path)
        df = feature_engineering(df)
        logger.success("Test data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return

    # Make predictions
    logger.info("Making predictions...")
    # Load the model using MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # Prepare features for prediction (drop unnecessary columns)
    features = df.drop(columns=["id", "species"])

    # Make predictions with the logged model
    predictions = model.predict(features)
    accuracy = model.score(features.values, df.species)
    logger.info(f"The accuracy in the predictions is {accuracy}")

    # Save predictions to a CSV file
    logger.info(f"Saving predictions to {predictions_path}...")
    try:
        pd.DataFrame(predictions, columns=["predicted_species"]).to_csv(
            predictions_path, index=False
        )
        logger.success("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")

    # Log predictions as an artifact
    mlflow.log_artifact(predictions_path)
    # End MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    app()
