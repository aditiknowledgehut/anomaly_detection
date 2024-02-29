#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc


# In[2]:


# Step 1: Collect the Time Series Data
data = pd.read_csv('AnomaData.csv')


# In[3]:


# Step 2: Exploratory Data Analysis (EDA)
# Data quality check
print(data.info())
print(data.describe())


# In[4]:


# Treat missing values
data.dropna(inplace=True)  # Drop rows with missing values for simplicity


# In[5]:


# Treat outliers using z-score method
def treat_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    data_filtered = data[z_scores < threshold]
    return data_filtered


# In[6]:


# Visualize the data before and after treating outliers
def visualize_data(data, title):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, kde=True)
    plt.title(title)
    plt.show()


# In[7]:


# Visualize the target variable before outlier treatment
visualize_data(data['y'], title='Before Outlier Treatment')


# In[8]:


# Treat outliers using z-score method
data_filtered_zscore = treat_outliers_zscore(data['y'])


# In[9]:


# Visualize the target variable after outlier treatment
visualize_data(data_filtered_zscore, title='After Outlier Treatment (Z-score Method)')


# In[11]:


# Step 3: Get the Correct Datatype for Date
data['time'] = pd.to_datetime(data['time'])


# In[14]:


# Extract the target variable 'y'
y = data['y']

# Drop the 'time' and 'y' columns from the dataset as they are not predictors
data.drop(columns=['time', 'y'], inplace=True)

# Feature Engineering
# Example 1: Calculating the mean of the 'x' columns
data['x_mean'] = data.mean(axis=1)

# Example 2: Calculating the standard deviation of the 'x' columns
data['x_std'] = data.std(axis=1)

# Example 3: Calculating the maximum value of the 'x' columns
data['x_max'] = data.max(axis=1)

# Example 4: Calculating the minimum value of the 'x' columns
data['x_min'] = data.min(axis=1)

# Example 5: Calculating the range of the 'x' columns
data['x_range'] = data['x_max'] - data['x_min']

# Example 6: Calculating the sum of the 'x' columns
data['x_sum'] = data.sum(axis=1)

# Example 7: Calculating the median of the 'x' columns
data['x_median'] = data.median(axis=1)

# Example 8: Calculating the skewness of the 'x' columns
data['x_skew'] = data.skew(axis=1)

# Example 9: Calculating the kurtosis of the 'x' columns
data['x_kurtosis'] = data.kurtosis(axis=1)

# Example 10: Calculating the variance of the 'x' columns
data['x_var'] = data.var(axis=1)

# Save the modified dataset
data.to_csv('modified_anomadata.csv', index=False)


# In[20]:


# Define the feature matrix X
X = data

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Isolation Forest
iforest = IsolationForest(contamination=0.1, random_state=42)
iforest.fit(X_train)

# Predict outliers
y_pred_iforest_train = iforest.predict(X_train)
y_pred_iforest_test = iforest.predict(X_test)


# In[24]:


# Convert predictions to binary labels (1 for anomaly, -1 for normal)
y_pred_iforest_train_binary = np.where(y_pred_iforest_train == -1, 1, 0)
y_pred_iforest_test_binary = np.where(y_pred_iforest_test == -1, 1, 0)


# In[25]:


# Step 8: Model Evaluation
# Evaluate the model using chosen metrics
accuracy = accuracy_score(y_test, y_pred_iforest_test_binary)
precision = precision_score(y_test, y_pred_iforest_test_binary)
recall = recall_score(y_test, y_pred_iforest_test_binary)
f1 = f1_score(y_test, y_pred_iforest_test_binary)
roc_auc = roc_auc_score(y_test, y_pred_iforest_test_binary)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


# In[27]:


pip install sagemaker


# In[1]:


import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearnModel


# In[29]:


pip install --upgrade botocore

