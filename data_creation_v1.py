#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os


# In[2]:


INPUT_PATH = '../input1'


# In[3]:


os.makedirs(f'{INPUT_PATH}/train/defog_np')
os.makedirs(f'{INPUT_PATH}/train/tdcsfog_np')


# In[4]:


events = pd.read_csv(f'{INPUT_PATH}/events.csv')
events = events[~events.Type.isnull()]


# In[5]:


defog = pd.read_csv(f'{INPUT_PATH}/defog_metadata.csv')
defog = defog[defog.Id.isin(events.Id)]


# In[6]:


g0=9.80665


# In[7]:


for i,r in defog.iterrows():
    data = pd.read_csv(f'{INPUT_PATH}/train/defog/{r.Id}.csv')
    sig = data[[ 'AccV', 'AccML', 'AccAP']].values
    target = data[['StartHesitation', 'Turn', 'Walking']].values
    
    sig = sig*g0
    np.save(f'{INPUT_PATH}/train/defog_np/{r.Id}_sig.npy',sig)
    np.save(f'{INPUT_PATH}/train/defog_np/{r.Id}_tgt.npy',target)
    


# In[8]:


data.columns


# In[9]:


tdcsfog = pd.read_csv(f'{INPUT_PATH}/tdcsfog_metadata.csv')
tdcsfog = tdcsfog[tdcsfog.Id.isin(events.Id)]


# In[11]:


for i,r in tdcsfog.iterrows():
    data = pd.read_csv(f'{INPUT_PATH}/train/tdcsfog/{r.Id}.csv')
    sig = data[[ 'AccV', 'AccML', 'AccAP']].values
    target = data[['StartHesitation', 'Turn', 'Walking']].astype(np.float32).values
    
    sig = sig
    sig_resample = []
    
    for i in range(3):
        sig_resample.append(librosa.resample(sig[:,i],orig_sr=128,target_sr=100))
    sig = np.stack(sig_resample,axis=1)
    
        
    target_resample = []
    for i in range(3):
        target_resample.append(librosa.resample(target[:,i],orig_sr=128,target_sr=100))
        
    target = np.stack(target_resample,axis=1)  
    np.save(f'{INPUT_PATH}/train/tdcsfog_np/{r.Id}_sig.npy',sig)
    np.save(f'{INPUT_PATH}/train/tdcsfog_np/{r.Id}_tgt.npy',target)
    


# In[ ]:




