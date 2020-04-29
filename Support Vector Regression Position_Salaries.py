#!/usr/bin/env python
# coding: utf-8

# In[61]:


#import libabry

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# In[62]:


#read dataset
dataset = pd.read_csv('~/Downloads/Data Science/data set/Position_Salaries.csv')
X = pd.DataFrame(dataset, columns = ['Gender', 'Age'])
y = pd.DataFrame(dataset, columns = ['EstimatedSalary'])


# In[68]:


#Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# In[69]:


#build model
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# In[ ]:


#predicte new value

y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred) 
view raw


# In[70]:


# Visualize SVR
X_grid = np.arange(min(X), max(X), 0.01) #this step required because data is feature scaled.
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




