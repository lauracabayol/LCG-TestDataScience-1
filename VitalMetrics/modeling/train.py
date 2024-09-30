from VitalMetrics.classifier import Classifier
from VitalMetrics.config import MODELS_DIR, PROCESSED_DATA_DIR, MODEL_PARAMS, FIGURES_DIR
from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")


app = typer.Typer()


@app.command()
def main(
    features_train_path: Path = PROCESSED_DATA_DIR / "penguin_train_features.csv",
    features_test_path: Path = PROCESSED_DATA_DIR / "penguin_test_features.csv",
    model_type: str = "RandomForest",
):
    # Load data
    logger.info(f"Loading training features from {features_train_path} and test from {features_test_path}...")
    try:
        df_train = pd.read_csv(features_train_path, header=0, sep=',')
        df_test = pd.read_csv(features_test_path)
        logger.success("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Split data
    logger.info("Splitting the data into training features and labels...")
    X_train, y_train = df_train.drop(columns=['id', 'species']).values, df_train.species.values
    X_test, y_test = df_test.drop(columns=['id', 'species']).values, df_test.species.values

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

        # Evaluate and save confusion matrix
        predictions = classifier.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        confusion_matrix_path = FIGURES_DIR / f'confusion_matrix_{model_type}.png'
        plt.savefig(confusion_matrix_path)

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
            sk_model=classifier,
            artifact_path="model",
            registered_model_name='PENGUINS_CLASSIFIER'
        )

        # Save the model locally (optional)
        filename = f"{model_type}_model.pkl"
        pickle.dump(classifier, open(MODELS_DIR / filename, "wb"))


if __name__ == "__main__":
    app()
