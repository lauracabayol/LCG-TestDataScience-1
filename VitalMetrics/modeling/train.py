from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn

from VitalMetrics.classifier import Classifier
from VitalMetrics.config import RAW_DATA_DIR, MODEL_PARAMS
from VitalMetrics.features import feature_engineering

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = typer.Typer()

@app.command()
def main(
    training_data_path: Path = RAW_DATA_DIR / "palmer-penguins-dataset-for-eda/penguins_train.csv",
    testing_data_path: Path = RAW_DATA_DIR / "palmer-penguins-dataset-for-eda/penguins_test.csv",
    model_type: str = "RandomForest",
):
    # Load data
    logger.info(
        f"Loading training and test data from {training_data_path} and {testing_data_path}..."
    )
    try:
        df_train = pd.read_csv(training_data_path)
        df_test = pd.read_csv(testing_data_path)
        logger.success("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Split data
    logger.info("Preprocessing data...")
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)
    logger.info("Selecting features and labels from training and evaluating...")
    X_train, y_train = df_train.drop(columns=["id", "species"]).values, df_train.species.values
    X_test, y_test = df_test.drop(columns=["id", "species"]).values, df_test.species.values

    # MLflow Tracking
    with mlflow.start_run() as run:

        logger.info(f"Training {model_type} model...")
        # Initialize the classifier
        classifier = Classifier(model_type=model_type)

        # Train the model
        classifier.train(X_train, y_train)

        # Evaluate the model
        accuracy = classifier.score(X_test, y_test)
        logger.success(f"Model training complete. {model_type} accuracy: {accuracy:.4f}")

        # Log model parameters
        mlflow.log_param("model_type", model_type)
        if model_type == "RandomForest":
            mlflow.log_param("n_estimators", MODEL_PARAMS["n_estimators"])
            mlflow.log_param("max_depth", MODEL_PARAMS["max_depth"])
        elif model_type == "GradientBoosting":
            mlflow.log_param("learning_rate", MODEL_PARAMS["learning_rate"])
            mlflow.log_param("n_estimators", MODEL_PARAMS["n_estimators"])
        elif model_type == "SVM":
            mlflow.log_param("C", MODEL_PARAMS["C"])
            mlflow.log_param("kernel", MODEL_PARAMS["kernel"])
        elif model_type == "KNN":
            mlflow.log_param("n_neighbors", MODEL_PARAMS["n_neighbors"])

        # Log accuracy metric
        mlflow.log_metric("accuracy", accuracy)

        # Log and register the model with MLflow
        mlflow.sklearn.log_model(
            sk_model=classifier, artifact_path="model", registered_model_name="PENGUINS_CLASSIFIER"
        )

if __name__ == "__main__":
    app()
