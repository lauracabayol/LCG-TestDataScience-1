from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from loguru import logger

from VitalMetrics.config import MODEL_PARAMS


class Classifier:
    def __init__(self, model_type: str = "RandomForest"):
        """Initialize the classifier based on the selected model type.

        Parameters:
        model_type (str): The type of classifier to initialize ('RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM', 'KNN').
        """
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=MODEL_PARAMS.get("n_estimators", 100),
                max_depth=MODEL_PARAMS.get("max_depth", None),
                random_state=MODEL_PARAMS.get("random_state", 42),
            )
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(
                n_estimators=MODEL_PARAMS.get("n_estimators", 100),
                learning_rate=MODEL_PARAMS.get("learning_rate", 0.1),
                max_depth=MODEL_PARAMS.get("max_depth", 3),
                random_state=MODEL_PARAMS.get("random_state", 42),
            )
        elif model_type == "LogisticRegression":
            self.model = LogisticRegression(
                solver=MODEL_PARAMS.get("solver", "lbfgs"),
                max_iter=MODEL_PARAMS.get("max_iter", 100),
                random_state=MODEL_PARAMS.get("random_state", 42),
            )
        elif model_type == "SVM":
            self.model = SVC(
                C=MODEL_PARAMS.get("C", 1.0),
                kernel=MODEL_PARAMS.get("kernel", "rbf"),
                gamma=MODEL_PARAMS.get("gamma", "scale"),
                random_state=MODEL_PARAMS.get("random_state", 42),
            )
        elif model_type == "KNN":
            self.model = KNeighborsClassifier(
                n_neighbors=MODEL_PARAMS.get("n_neighbors", 5),
                algorithm=MODEL_PARAMS.get("algorithm", "auto"),
            )
        else:
            logger.error(f"Model type {model_type} not recognized. Defaulting to RandomForest.")
            self.model = RandomForestClassifier(
                n_estimators=MODEL_PARAMS.get("n_estimators", 100),
                max_depth=MODEL_PARAMS.get("max_depth", None),
                random_state=MODEL_PARAMS.get("random_state", 42),
            )

        logger.info(f"{model_type} classifier initialized.")

    def train(self, X_train, y_train):
        """Train the model on the given training data."""
        logger.info("Training the classifier...")
        self.model.fit(X_train, y_train)
        logger.success("Model training complete.")

    def predict(self, X_test):
        """Make predictions on the test data."""
        logger.info("Making predictions on the test data...")
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        """Return the accuracy of the model on the test data."""
        logger.info("Calculating model accuracy on the test set...")
        return self.model.score(X_test, y_test)
