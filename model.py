# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:23:43 2020

@author: Admin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("D:\Data science\Assignments docs\Multi linear Regression\Computer_Data.csv")

dataset = dataset.drop(columns=["Unnamed: 0","cd", "multi","premium"], axis=1)

X = dataset.iloc[:,1:7]
y = dataset.iloc[:,0]


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
