# Aldehyde Dehydrogenase (ALDH1) Inhibitor Prediction Project

## Overview

The goal of this project is to predict the top 100 molecules that can inhibit Aldehyde dehydrogenase (ALDH1). This is an enzyme involved in the detoxification of aldehydes produced by alcohol metabolism and is implicated in the development of certain types of cancers.

In the first phase of the project, we use a dataset containing molecules labeled either as inhibitors (1) or non-inhibitors (0) of ALDH1. This labeled data is used to train a Machine Learning model that can predict whether a given molecule can inhibit ALDH1 or not.

The second phase involves applying this trained model to a new dataset of molecules (without inhibition information) to predict the top 100 potential ALDH1 inhibitors. 

## Data

The data used in this project are sets of molecules, with each molecule having associated properties and characteristics. In the initial training phase, each molecule is labeled with either a 0 (no inhibition) or a 1 (inhibition). 

For the prediction phase, we utilize a dataset of molecules without known inhibition status. The task is to predict the inhibition status and select the top 100 inhibitors.

## Methodology

1. **Data Preprocessing**: Initially, we perform some necessary preprocessing steps to clean and format the data.

2. **PCA**: We then perform Principal Component Analysis (PCA) to reduce the dimensionality of the dataset and to identify the principal components that capture the most variance in the data.

3. **Model Training**: Post PCA, we proceed to train a Machine Learning model using the processed data. This model will learn from the labeled data to differentiate between inhibitors and non-inhibitors.

4. **Prediction and Selection**: The trained model is then applied to the second dataset to predict potential inhibitors. The top 100 molecules predicted as inhibitors are selected as the final output.

## Repository Contents

The main components of the repository are as follows:

- `data/`: This folder contains the datasets used for training and prediction.

- `notebooks/`: This folder contains Jupyter notebooks for data preprocessing, PCA, model training, prediction, and other analyses.

- `src/`: This folder contains Python scripts for various stages of the project.

- `results/`: This folder holds the final results, including the list of top 100 predicted inhibitors.

- `requirements.txt`: This file lists the Python dependencies required for this project.

- `README.md`: This file provides an overview of the project and repository.

## Contributors

- Tijmen Vierwind
- Kay Janssen
- Stan Dobbelsteen
- Giel Dobbelsteen
- Tristan Muir
- Tim Stassen
- Sjoerd de Ruijter




