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

# # EXPLORATORY DATA ANALYSIS (EDA)

# This notebook does EDA on [this kaggle dataset](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda/data).
#

# The dataset contains features of Penguins collected for research that was conducted as part of the Palmer Station, Antarctica.

# The raw data is saved at the /data/raw directory. The script */scripts/download_data.py* downlaods it directly form kaggle. 

# #### Python imports

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set global settings for all plots
plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "serif"
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from VitalMetrics.config import FIGURES_DIR
# -

# ## 1. LOAD RAW DATA

# We first load all the dataset. For trainig/testing proposes, we will later split the data.

df = pd.read_csv("../data/raw/palmer-penguins-dataset-for-eda/penguins.csv")

print(f"Dataset size = {df.shape}")

df.head()

# ## 2. CLEAN DATA 

# #### 2.1 Check if there are rows with missing values or repeated entries

# There are missing values (NaNs), but there are not duplicated entries. 
# For convenience, we get rid of all entries with null values

df.isnull().values.any()

df.dropna(inplace=True)

df.duplicated().any()

# #### 2.2. Which columns do we have and which ones could be relevant?

# - id: Not relevant for the classification
# - species: The class we will predict. There are three classes: ['Adelie', 'Gentoo', 'Chinstrap']. Needs to be encoded.
# - island: Relevant. Thre possibilities. ['Torgersen', 'Biscoe', 'Dream']. Needs to be encoded.
# - bill_length_mm: Could be relevant. Check dependencies.
# - bill_depth_mm: Could be relevant. Check dependencies.
# - flipper_length_mm: Could be relevant. Check dependencies. 
# - body_mass_g: Could be relevant. Check dependencies. 
# - sex: Could be relevant. Needs to be encoded. Check dependencies. 
# - year: It could be relevant if species evolved with time. Check. Three years, if important the variable proves important, we need to scale it.

# #### 2.3. Label encoding the selected columns

columns_to_encode = ["species", "sex", "island"]

LabEnc_mapping = {}
for col in columns_to_encode:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col].values)
    LabEnc_mapping[col] = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

# ## 3. LOOK AT CORRELATIONS BETWEEN THE DIFFERENT VARIABLES

print(FIGURES_DIR / "CorrelationFeatures.png")

plt.figure(figsize=(20, 10))
sns.heatmap(df.drop(columns=["id"]).corr(), annot=True)
plt.savefig(FIGURES_DIR / "CorrelationFeatures.png", bbox_inches="tight")

# The specie type strongly correlates with: 
# - island
# - bill length
# - bill depth
# - flipped length
# - body mass

# No correlation with sex and year. We will further test dependencies between these two variables and the specie and if this is confirmed, we will not use them for the classification.

# ### 3.1 Demographics

# We first test if the three species have a balanced gender distribution. We can see that this is indeed the case, which further indicates that gender does not provide relevant information for the classification.

# +
plt.figure(figsize=(10, 6))

sns.countplot(x="sex", hue="species", data=df, palette="deep")

plt.xlabel("Gender")
plt.ylabel("Number of Penguins")

plt.xticks([0, 1], ["Male", "Female"])

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = ["Adelie", "Chinstrap", "Gentoo"]

plt.legend(handles, new_labels)

plt.savefig(FIGURES_DIR / "PenguinsGenderHist.png", bbox_inches="tight")
plt.show()
# -

# We can apply a similar analysis to the island variable. In this case, it's evident that the island plays a significant role in distinguishing penguin species. For example, the Gentoo penguin is exclusively found on Biscoe Island, which indicates that the island variable can serve as a unique identifier for certain penguin species, making it a highly relevant feature for species classification.

# This observation is particularly useful in our context, but it also highlights a potential issue: if our dataset is not fully representative, the absence of certain species on specific islands could introduce bias into our model. For instance, if Gentoo penguins are underrepresented or missing from the data for Biscoe Island, the model might fail to accurately predict their presence, leading to skewed results and reduced generalizability.

# +
plt.figure(figsize=(10, 6))

sns.countplot(x="island", hue="species", data=df, palette="deep")

plt.xlabel("Island")
plt.ylabel("Number of Penguins")

plt.xticks([0, 1, 2], ["Biscoe", "Dream", "Torgersen"])

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = ["Adelie", "Chinstrap", "Gentoo"]

plt.legend(handles, new_labels)

plt.savefig(FIGURES_DIR / "PenguinsIslandHist.png", bbox_inches="tight")
plt.show()
# -

# We can also check the population evolution over time. There is no clear trend, which explain the lack of correlation. We will not use this variable for prediction either.

# +
penguin_counts = df.groupby(["year", "species"]).size().reset_index(name="count")

plt.figure(figsize=(12, 6))  # Set figure size
sns.lineplot(data=penguin_counts, x="year", y="count", hue="species", marker="o")

plt.title("Penguin Population Evolution Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Recorded Penguins")

handles, labels = plt.gca().get_legend_handles_labels()
new_labels = ["Adelie", "Chinstrap", "Gentoo"]

plt.legend(handles, new_labels)
plt.xticks([2007, 2008, 2009])
plt.savefig(FIGURES_DIR / "PenguinsEvoTime.png", bbox_inches="tight")

plt.show()
# -

# Class imbalance can indeed be a challenge for predictive modeling, especially in classification tasks. In the case of the penguin dataset, the underrepresentation of Chinstrap penguins compared to other species could lead to biased model performance. The model might become skewed towards predicting the more frequent classes. We'll need to assess whether this imbalance impacts the model's ability to generalize across species. 

# +
plt.figure(figsize=(8, 6))  # Set figure size
sns.countplot(x="species", data=df, palette="deep")
plt.xlabel("Species")
plt.ylabel("Counts")
plt.xticks([0, 1, 2], ["Adelie", "Chinstrap", "Gentoo"])
plt.savefig(FIGURES_DIR / "PenguinsSpecies.png", bbox_inches="tight")

plt.show()
# -

# ## 4. CLUSTERING

independent_variables = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
dependent_variable = ["species"]

# +
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X = df[independent_variables].values
clustering_estimator = KMeans(n_clusters=3, max_iter=100, random_state=64)
clustering_class = clustering_estimator.fit(X)
# -

cm = confusion_matrix(df.species.values, clustering_class.labels_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# A simple kmeans algorithm can already make a reasonable classification
