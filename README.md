# VitalMetrics

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

 This repository uses data from the open Penguin data set available [here](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda).

The repository documentation can be found [here](https://lauracabayol.github.io/LCG-TestDataScience-1/).

The repository explores four algoritms: 

- Gradient Boosting
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

After testing, the LSTM model was selected as the best performer and has been deployed for production.

## Project Organization

```
├── LICENSE                                            <- Open-source MIT license 
├── README.md                                          <- The top-level README for developers using this project.
├── data
│   └── raw/palmer-penguins-dataset-for-eda            <- The original data
│   └── processed/                                     <- Data features ready for training models (train and test csv files)
│
├── docs                                               <- Mkdocs project (deployed to GitHub)
│
├── scripts                                            <- additional scripts not integrated in the main project
│──
    ├── download_data.py                               <- script to download data from kaggle
│
├── notebooks                                          <- Jupyter notebooks. 
│──
    ├── EDA.py                                         <- Notebook with exploratory data analysis
    ├── Make_predictions.py                            <- Notebook to make predictions and compare models registered at MLFlow. Only accessible if one has the MLFLow logs (not uploaded to GH)
    ├── HTML_notebooks             <- Stable HTML version of notebooks
│
├── pyproject.toml                                     <- Project configuration file with package metadata  
│
├── reports                                            <- LaTeX report
│   └── LCG_TestDataScience_1.pdf                      <- Generated LaTeX report
│
├── requirements.txt                                   <- The requirements file for reproducing the analysis 
│
└── VitalMetrics                                       <- Source code for use in this project.
    │
    ├── __init__.py                                    <- Makes VitalMetrics a Python module
    │
    ├── config.py                                      <- Store useful variables and configuration
    │    
    ├── features.py                                    <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py                                 <- Code to run model inference with trained models          
    │   └── train.py                                   <- Code to train models
    |
    |__ classifier.py                                  <- Main model code
│
├── app                                                <- Directory with notebook to run on a docker container
│   └── data                                           <- Directory with data required to make predictions
    └── ML_artifacts                                   <- Directory with model artifacts from MLFlow
    └──Make_predictions.ipynb                          <- Notebook to make predictions
    └──requirements.txt                                <- Required Python modules
├
├── app.py                                             <-Script for model integration in weights and biases.
├── model.pkl                                          <-Model for integration in weights and biases
├── Dockerfile                                         <-Code to launch the docker container
```

## Installation

### Installation and Environment Setup
We recommend using a virtual environment to install the project dependencies and maintain an isolated workspace.
#### Setting Up a Virtual Environment:
To create a virtual environment using <venv>, run the following commands:
```bash
python -m venv venv
source venv/bin/activate   
```
#### Setting Up a Conda Environment:
Alternatively, you can create a Conda environment with Python 3.10 by executing the following commands:
```
conda create -n TempForecast -c conda-forge python=3.10
conda activate TempForecast
```
The required python modules are in the <requirements.txt> file.

You will also need to clone the repository to your local environment by executing the following commands:

```bash
git clone https://github.com/lauracabayol/LCG-TestDataScience-1.git
cd LCG-TestDataScience-1/
```
and then install it.

```
pip install -e .
``` 

### Deployed Gradient Boosting model
**The available deployed model as of today is the Gradient Boosting**. There are two different ways of accessing the deployed model:

#### Through weights and biases:
We have deployed the model in the WaB platform and it is available [here](https://huggingface.co/spaces/lauracabayol/PENGUINS_CLASSIFIER). This allows the user to make single predictions from a set of features. It is publicly available for everyone and userfriendly, but does not support making predictions for more than one sample simultaneously. 

#### Model in a Docker container:

For convenience, we have created a Docker container that includes the Gradient Boosting model along with all necessary dependencies. This allows you to run the model without needing access to MLflow or the associated logs.

##### Instructions to Run the Docker Container:

**Build the Docker Image**: First, clone the repository and navigate to the project directory. Then, build the Docker image:
```bash
docker build -t penguin-classificaton:latest .
```
**Run the Docker Container**: Once the image is built, run the container using the following command:
```
docker run -p 9999:9999 penguin-classificaton:latest
```
This will start a Jupyter notebook where you can interact with the pre-trained best-performing model.

**Access the Jupyter Notebook**: Open your web browser and go to:
```bash
http://localhost:9999
```
The notebook is pre-configured to load and run the best model, so you can use it without needing to access MLflow.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as long as you adhere to the license terms.






