#!/usr/bin/env python
# coding: utf-8

# # Titanic Prediction
# ## We want to know which passenger are most likely to survive using machine learning
# 

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.ensemble import RandomForestRegressor


# In[2]:


#importing data
folder_test = "\\Users\\Lorenzo de Sario\\Desktop\\Titanic prediction\\test.csv"
folder_train = "\\Users\\Lorenzo de Sario\\Desktop\\Titanic prediction\\train.csv"
train = pd.read_csv(folder_train)
test = pd.read_csv(folder_test)
train.head()


# In[3]:


train.info()


# In[4]:


#explore outliers
train.plot(kind = 'box', subplots = 1, figsize = (40,10))
plt.show()


# In[5]:


#analyze effects of some variables against survived status
pp = sns.pairplot(data=train,
                  y_vars=['Survived'],
                  x_vars=["Age", "Sex", "SibSp", "Fare"])


# In[6]:


model = RandomForestRegressor(n_estimators = 100, max_depth = 5, random_state = 1)
observables = ["Pclass", "Sex", "SibSp", "Parch"]
x = train[observables]
y = train['Survived']
X = pd.get_dummies(x)
X_test = pd.get_dummies(test[observables])


# In[8]:


model.fit(X, y)
morti = model.predict(X_test)
print('predictionssssssss\n', morti)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': morti})


# In[12]:


output.head(30)


# In[ ]:




