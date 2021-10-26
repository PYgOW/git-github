#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[7]:


df.head(5)


# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


scaler=StandardScaler()
scaler.fit(df)


# In[11]:


scaled_data=scaler.transform(df)


# In[12]:


scaled_data


# In[13]:


from sklearn.decomposition import PCA


# In[14]:


pca=PCA(n_components=2)


# In[15]:


pca.fit(scaled_data)


# In[16]:


x_pca=pca.transform(scaled_data)


# In[17]:


scaled_data.shape


# In[18]:


x_pca.shape


# In[19]:


scaled_data


# In[20]:


x_pca


# In[21]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


# In[ ]:




