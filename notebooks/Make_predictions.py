# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: capgemini
#     language: python
#     name: capgemini
# ---

# # PREDICTING PENGUIN CLASS FROM DEPLOYED MODEL

# In this notebook, we predict the penguin spice based on their:
# - island
# - bill length
# - bill depth
# - flipped length
# - body mass

# We will use a Gradient Boosting algorithm that we have registered in MLFlow

# #### Python imports

# +
import requests
import pandas as pd
from pathlib import Path
import mlflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

from VitalMetrics.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from VitalMetrics.features import feature_engineering
# -

# #### Define parameters

#test_data_path = Path(PROCESSED_DATA_DIR /"penguin_test_features.csv")
test_data_path = Path(RAW_DATA_DIR / "palmer-penguins-dataset-for-eda/penguins_test.csv")
model_type = "Gradient Boosting"

# ## PREDICTIONS WITH DEPLOYED MODEL

# ### Retrieving the model

model_name = "PENGUINS_CLASSIFIER"
model_version = 16
model_uri = f"models:/{model_name}/{model_version}"
deployed_model = mlflow.sklearn.load_model(model_uri)

# ### Opening test data

df = pd.read_csv(test_data_path)
df = feature_engineering(df)
features = df.drop(columns=['id', 'species']).values.tolist()  
true_class = df.species.values

penguin_predicted_species = deployed_model.predict(features)

# ### EVALUATE PREDICTIONS

cm = confusion_matrix(true_class, penguin_predicted_species)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


