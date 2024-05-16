# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # **Titanic Kaggle Competition Submission**

# %% [markdown]
# **Check and install the dependencies**

# %%
# !curl -sSL https://raw.githubusercontent.com/K14aNB/Titanic-Kaggle-Competition/main/requirements.txt

# %%
# Run this command in terminal before running this notebook as .py script
# Installs dependencies from requirements.txt present in the repo
# %%capture
# !pip install -r https://raw.githubusercontent.com/K14aNB/Titanic-Kaggle-Competition/main/requirements.txt

# %% [markdown]
# **Import the libraries**

# %%
import mlflow
import pandas as pd
import env_setup
import os

# %% [markdown]
# **Environment Setup**

# %%
# Setup Environment(Downloading data and setting output formats specified in config.yaml)
result_path=env_setup.setup(repo_name='Titanic-Kaggle-Competition',nb_name='Titanic-Kaggle-Competition-Submission')

# %% [markdown]
# **Get Artifact from MLFlow**

# %%
mlflow.search_runs(experiment_names=['Titanic-Kaggle-Competition'])

# %% [markdown]
# **Read the data**

# %%
X_test=pd.read_csv(os.path.join(result_path,'test.csv'))

# %%
logged_model='runs:/195fb6f7f28a4b20b128e005a5617119/Logistic Regression_CV'

# %%
# Load model as a PyFuncModel
loaded_model=mlflow.pyfunc.load_model(logged_model)

# %%
loaded_model(X_test)

# %%
