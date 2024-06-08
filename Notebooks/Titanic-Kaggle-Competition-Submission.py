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
# # **Titanic Kaggle Competition - Submission**

# %% [markdown]
# **Check and install the dependencies**

# %%
# !curl -sSL https://raw.githubusercontent.com/K14aNB/titanic-kaggle-competition/main/requirements.txt

# %%
# Run this command in terminal before running this notebook as .py script
# Installs dependencies from requirements.txt present in the repo
# %%capture
# !pip install -r https://raw.githubusercontent.com/K14aNB/titanic-kaggle-competition/main/requirements.txt

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
result_path=env_setup.setup(repo_name='titanic-kaggle-competition',nb_name='Titanic-Kaggle-Competition-Submission')

# %% [markdown]
# **Retrieve Runs from MLFlow**

# %%
runs=mlflow.search_runs(experiment_names=['Titanic-Kaggle-Competition'])

# %%
runs.head()

# %% [markdown]
# **Get Model from run_id**

# %%
logged_model=f'runs:/{runs.loc[0,"run_id"]}/{runs.loc[0,"tags.mlflow.runName"]}'

# %% [markdown]
# **Read the data**

# %%
X_test=pd.read_csv(os.path.join(result_path,'test.csv'))

# %%
X_test.head()

# %%
X_test.info()

# %% [markdown]
# **Load the model from MLFlow**

# %%
# Load model as a PyFuncModel
loaded_model=mlflow.sklearn.load_model(logged_model)

# %% [markdown]
# **Model Predictions**

# %%
predictions=loaded_model.predict(X_test)

# %%
output=pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':predictions})

# %%
output.head()

# %%
output.to_csv('submission.csv',index=False)

# %%
# !kaggle competitions submit -f submission.csv -m 'DecisionTree_v0.8' titanic

# %%
