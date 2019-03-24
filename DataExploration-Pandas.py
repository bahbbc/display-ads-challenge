#!/usr/bin/env python
# coding: utf-8

# # Sample EDA

# In[ ]:


import dask.dataframe as dd
import pandas as pdz
pd.set_option('display.max_columns', 500)
import lightgbm as lgb
from tqdm import tqdm


# Since the dataset doesn't have a header we will use the dataset README to create a preliminary header

# In[9]:


column_names = ['label','interger1','interger2','interger3','interger4','interger5','interger6',
                'interger7','interger8','interger9','interger10','interger11','interger12','interger13',
                'categorical1','categorical2','categorical3','categorical4','categorical5','categorical6',
                'categorical7','categorical8','categorical9','categorical10','categorical11','categorical12',
                'categorical13','categorical14','categorical15','categorical16','categorical17','categorical18',
                'categorical19','categorical20','categorical21','categorical22','categorical23','categorical24',
                'categorical25','categorical26']


# In[10]:


df = pd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/train.txt', sep='\t', 
            names=column_names, chunksize=7, engine='python')


# In[25]:


answer = pd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/test.txt', 
                     sep='\t', names=column_names, chunksize=7, engine='python')


# In[19]:


from sklearn.linear_model import SGDClassifier

logistic = SGDClassifier(loss='log', 
                         penalty='l2', 
                         alpha=0.1, 
                         fit_intercept=False, 
                         max_iter=5, 
                         shuffle=True, 
                         verbose=1, 
                         n_jobs=1,
                         random_state=42, 
                         learning_rate='optimal',
                         tol=0.01)


# In[ ]:


for chunk in df:
    integer_cols = ['interger1', 'interger2', 'interger3', 'interger4',
       'interger5', 'interger6', 'interger7', 'interger8', 'interger9',
       'interger10', 'interger11', 'interger12', 'interger13']
    X = chunk[integer_cols]
    y = chunk['label']
    
    X = X.fillna(0)
    
    logistic.partial_fit(X, y, classes=[0,1])
    print('Score:', logistic.score(X, y))


# In[9]:




