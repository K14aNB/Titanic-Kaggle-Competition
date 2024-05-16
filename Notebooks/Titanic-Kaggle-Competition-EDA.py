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
# # **Titanic-Kaggle-Competition-EDA**

# %% [markdown]
# ## **Data Dictionary**
#

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import mlflow
import env_setup
import os

# %% [markdown]
# **Environment Setup**

# %%
result_path=env_setup.setup(repo_name='Titanic-Kaggle-Competition',nb_name='Titanic-Kaggle-Competition-EDA')

# %% [markdown]
# **Read the data**

# %%
titanic_train=pd.read_csv(os.path.join(result_path,'train.csv'))

# %%
titanic_train.head()

# %%
titanic_train.info()

# %%
titanic_train.describe()

# %% [markdown]
# ### **Exploratory Data Analysis**

# %% [markdown]
# **Check for missing values**

# %%
titanic_train.isna().sum()

# %% [markdown]
# **Age column has lot of missing values.  
# Cabin column can be dropped due to significant percentage of missing values  
# Embarked column missing values can be imputed the mode**

# %%
# Inspect Embarked column
titanic_train['Embarked'].value_counts()

# %% [markdown]
# **Check for duplicated rows**

# %%
titanic_train.duplicated().sum().sum()

# %% [markdown]
# **Inspect `Fare` column**

# %%
fig=plt.figure(figsize=(10,5))
sns.boxplot(x='Fare',data=titanic_train)
plt.xlabel('Fare')
plt.title('Box plot for Fare')
plt.show()

# %% [markdown]
# **There seem to be outliers present in `Fare` column. We can keep this for now for the baseline model**

# %% [markdown]
# **Relationship between `Fare` and `Pclass`**

# %%
fig=plt.figure(figsize=(10,5))
sns.barplot(x='Pclass',y='Fare',hue='Survived',data=titanic_train)
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.title('Bar plot of Pclass vs Fare')
plt.show()

# %% [markdown]
# **Clearly the bar plot indicates that people from Higher Socio-economic background who paid higher fare had better Survival rate.  
# Fare and Pclass have strong relationship with Survival**

# %% [markdown]
# **Inspect `Parch` and `SibSp` columns**

# %%
titanic_train['Parch'].value_counts()

# %%
titanic_train['SibSp'].value_counts()

# %%
# Count plot of Parch
fig=plt.figure(figsize=(10,5))
sns.countplot(x='Parch',hue='Survived',data=titanic_train)
plt.xlabel('Parents/Children travelling along with Passenger')
plt.ylabel('Count')
plt.title('Count plot of Parents/Children')
plt.show()

# %%
# Count plot of SibSp
fig=plt.figure(figsize=(10,5))
sns.countplot(x='SibSp',hue='Survived',data=titanic_train)
plt.xlabel('Siblings/Spouses travelling along with Passenger')
plt.ylabel('Count')
plt.title('Count plot of Siblings/Spouses')
plt.show()

# %% [markdown]
# **Both `Parch` and `SibSp` show that people travelling alone are more in number and also their Survival rate is lower than people travelling with Family  
# Although the dataset mentions the value 0 is given for Children travelling with nannies, we can make reasonable assumption that people travelling alone (without family) are of category 0 in `Parch` and `SibSp`**
#

# %% [markdown]
# **We can possibly combine `Parch` and `SibSp` into a single column with binary values to indicate whether they are travelling alone or with family**

# %%
X_analysis=titanic_train.copy()
X_analysis.loc[(X_analysis['Parch']==0)|(X_analysis['SibSp']==0),'Travelling_Alone']=1
X_analysis.loc[(X_analysis['Parch']>0)|(X_analysis['SibSp']>0),'Travelling_Alone']=0

# %%
X_analysis['Travelling_Alone'].value_counts()

# %% [markdown]
# **Inspect Relation between `Sex` and `Travelling_Alone` columns based on Survival**

# %%
fig=plt.figure(figsize=(10,5))
sns.barplot(x='Sex',y='Travelling_Alone',hue='Survived',data=X_analysis)
plt.xlabel('Sex')
plt.ylabel('Travelling Alone')
plt.title('Bar plot of Sex vs Travelling Alone')
plt.show()

# %% [markdown]
# **From the bar plot, it clear that female passengers who were travelling alone had a higher survival rate while Male passengers with families had a higher Survival rate**

# %% [markdown]
# **Split Predictors and Target**

# %%
y=titanic_train.loc[:,'Survived']
X=titanic_train.drop(['Survived'],axis=1)

# %%
params={
    'num_impute_strategy':'median',
    'num_fill_value':'NA',
    'cat_impute_strategy':'most_frequent',
    'lr_penalty': 'l2',
    'random_state':0,
    'max_iter':500,
    'cv_folds':5
}

# %% [markdown]
# ### **Preprocessing**

# %%
# Columns to drop
cols_to_drop=['PassengerId','Name','Ticket','Cabin']

# %%
# Separate numerical and categorical columns
num_cols=[col for col in X.columns if X[col].dtype in ['int32','int64','float32','float64'] and col not in cols_to_drop]
cat_cols=[col for col in X.columns if X[col].dtype=='object' and col not in cols_to_drop]

# %%
print(num_cols)
print(cat_cols)

# %%
# Initialize SimpleImputer to impute missing values for numerical columns
if params.get('num_impute_strategy')=='constant':
    numerical_imputer=SimpleImputer(strategy=params.get('num_impute_strategy'),fill_value=params.get('num_fill_value'))
else:
    numerical_imputer=SimpleImputer(strategy=params.get('num_impute_strategy'))

# %% [markdown]
# **Numerical Pipeline Steps**

# %%
numerical_pipeline=Pipeline(steps=[
    ('numerical_imputer',numerical_imputer)
])

# %%
# Initialize SimpleImputer to impute missing values for categorical columns
categorical_imputer=SimpleImputer(strategy=params.get('cat_impute_strategy'))

# %% [markdown]
# **Encode categorical variables**

# %%
# Initialize OneHotEncoder to encode categorical variables
cat_encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)

# %% [markdown]
# **Categorical Pipeline Steps**

# %%
categorical_pipeline=Pipeline(steps=[
    ('categorical_imputer',categorical_imputer),
    ('categorical_encoder',cat_encoder)
])

# %% [markdown]
# **Combine Numerical and Categorical Pipelines**

# %%
# Combine Numerical and categorical Pipeline steps using ColumnTransformer
preprocessor=ColumnTransformer(transformers=[
    ('numerical_pipeline',numerical_pipeline,num_cols),
    ('categorical_pipeline',categorical_pipeline,cat_cols),
])


# %% [markdown]
# ### **Feature Extraction**

# %% [markdown]
# **Combine `Parch` and `SibSp` columns into a single column with binary values**

# %%
class FeatureCombiner(BaseEstimator,TransformerMixin):
    def __init__(self,columns,new_col):
        self.columns=columns
        self.new_col=new_col

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_=pd.DataFrame(X,columns=['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S'])
        X_.loc[(X_[self.columns[0]]==0)|(X_[self.columns[1]]==0),self.new_col]=1
        X_.loc[(X_[self.columns[0]]>0)|(X_[self.columns[1]]>0),self.new_col]=0
        return X_



# %%
# Initialize Custom Transformer FeatureCombiner
feature_combiner=FeatureCombiner(columns=['Parch','SibSp'],new_col='Travelling_Alone')


# %%
class SelectFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,columns):
        self.columns=columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X.loc[:,self.columns]


# %%
# Initialize Custom Transformer SelectFeatures
select_features=SelectFeatures(['Pclass','Age','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S','Travelling_Alone'])

# %% [markdown]
# **Feature Step**

# %%
feature_pipeline=Pipeline(steps=[
    ('feature_combiner',feature_combiner),
    ('select_features',select_features)
])

# %% [markdown]
# **Logistic Regression Model**

# %%
lr_run_name='Logistic Regression_CV'
if params.get('lr_penalty')=='None':
    lr=LogisticRegression(penalty=None,random_state=params.get('random_state'),max_iter=params.get('max_iter'))
else:
    lr=LogisticRegression(penalty=params.get('lr_penalty'),random_state=params.get('random_state'),max_iter=params.get('max_iter'))

# %% [markdown]
# ### **Pipeline**

# %% [markdown]
# **Define Pipeline Steps**

# %%
pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('features',feature_pipeline),
    ('model',lr)
])

# %% [markdown]
# ### **Cross-validation**

# %%
cv_scores=cross_validate(pipeline,X,y,scoring=['accuracy','precision','recall','roc_auc'],cv=params.get('cv_folds'))

# %% [markdown]
# **Model Metrics**

# %%
cv_scores

# %%
avg_accuracy=round(cv_scores['test_accuracy'].mean(),3)
avg_precision=round(cv_scores['test_precision'].mean(),3)
avg_recall=round(cv_scores['test_recall'].mean(),3)
avg_roc_auc=round(cv_scores['test_roc_auc'].mean(),3)

# %% [markdown]
# ### **Log the model in MLFlow**

# %%
with mlflow.start_run(run_name=lr_run_name):
    mlflow.log_params(params)
    mlflow.log_metrics({'accuracy':avg_accuracy,'precision':avg_precision,'recall':avg_recall,'roc_auc':avg_roc_auc})
    signature=mlflow.models.infer_signature(X,y,params)
    mlflow.sklearn.log_model(sk_model=pipeline,artifact_path=lr_run_name,signature=signature,registered_model_name=lr_run_name)


# %%
