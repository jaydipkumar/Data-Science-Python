#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# In[79]:


Emp_data


# In[80]:


Emp_data = pd.read_csv("~/Downloads/Data Science/data set/emp_data.csv")


# In[81]:


x = Emp_data.iloc[:, :-1].values 
y = Emp_data.iloc[:,-1:].values 


# In[64]:


x


# In[65]:


y


# In[30]:


Emp_data


# In[19]:


stats.probplot(Emp_data.Churn_out_rate, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()


# In[20]:


stats.probplot(Emp_data.Salary_hike, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()


# In[83]:


plt.plot(Emp_data.Churn_out_rate,Emp_data.Salary_hike) 
plt.show() 


# In[27]:


reg = LinearRegression()


# In[84]:


reg.fit(x,y)
print(reg.score(x, y)) 


# In[85]:


reg.fit(np.log(x),y)
print(reg.score(np.log(x), y)) 


# In[86]:


reg.fit(np.log(x),np.log(y))
print(reg.score(np.log(x),np.log(y)))

