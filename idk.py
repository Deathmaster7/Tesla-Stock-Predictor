#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from pandas_datareader import data as pdr

stock = pdr.get_data_yahoo('AAPL', start='2011-07-01', end = '2023-12-31')


# In[2]:


stock_show=stock


# In[3]:


stock = stock.drop('Adj Close', axis=1)


# In[4]:


stock.head()


# In[5]:


stock.info()


# In[6]:


stock.describe()


# In[7]:


stock.to_csv('applekidetail.csv')


# In[8]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib as plt
stock.hist(bins=50, figsize=(12,8))


# ## Train Test Splitting

# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(stock, test_size=0.2, random_state=42)
print(f"Rows in train: {len(train_set)}\nRows in test : {len(test_set)}")


# In[11]:


stock = train_set.copy()


# ## Looking for correlation

# In[12]:


corr_matrix = stock.corr()
corr_matrix['Close'].sort_values(ascending=False)


# In[13]:


from pandas.plotting import scatter_matrix
attributes = ["High", "Low"]
scatter_matrix(stock[attributes], figsize=(12, 8))


# ## Trying out attributes

# In[14]:


stock['OpenHigh']=stock['Open']/stock['High']


# In[15]:


corr_matrix = stock.corr()
corr_matrix['Close'].sort_values(ascending=False)


# In[16]:


stock = train_set.drop('Close', axis=1)
stock_labels = train_set['Close'].copy()


# ## Pipeline

# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[18]:


my_pipeline = Pipeline([
    ('stdscaler', StandardScaler())
])


# In[19]:


stock.head()


# In[20]:


stock_num_tr = my_pipeline.fit_transform(stock)


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = LinearRegression()
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()

model.fit(stock_num_tr, stock_labels)


# In[22]:


some_data = stock.iloc[:5]
some_labels = stock_labels.iloc[:5]


# In[23]:


some_data


# In[24]:


prepared_data = my_pipeline.transform(some_data)


# In[25]:


model.predict(prepared_data)


# ## Evaluating

# In[26]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, stock_num_tr, stock_labels, scoring='neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)


# In[27]:


def print_scores(scores):
    print('scores: ', scores)
    print('mean: ', scores.mean())
    print('Standard deviation: ', scores.std())

print_scores(rmse_scores)


# ## Saving the Model

# In[28]:


from joblib import dump, load
dump(model, "stock.joblib")


# ## Testing the Model

# In[29]:


from sklearn.metrics import mean_squared_error
x_test = test_set.drop("Close", axis=1)
y_test = test_set['Close'].copy()
x_test_prepared_data = my_pipeline.transform(x_test)
final = model.predict(x_test_prepared_data)
final_mse = mean_squared_error(y_test, final)
final_rmse = np.sqrt(final_mse)


# In[30]:


prepared_data


# ## Using the model

# In[42]:


features = np.array([[32.67,	32.74,	32.48,	123934000]])
df = pd.DataFrame(features, columns =['Open', 'High', 'Low', 'Volume'])
ig = my_pipeline.transform(df)
model.predict(ig)


# In[32]:


stock_labels.head()


# In[33]:


train_set


# In[34]:


import pickle


# In[35]:


data = {"model": model}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[36]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]


# In[39]:


x = my_pipeline.transform([[32.67, 32.74, 32.47, 123934000.00]])


# In[40]:


x


# In[41]:


#enable_gui


# In[ ]:




