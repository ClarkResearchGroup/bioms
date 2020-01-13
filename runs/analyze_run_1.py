#!/usr/bin/env python
# coding: utf-8

# # Read in the run data

# In[1]:


import pandas as pd

data_frame = pd.read_pickle('output_run_1.p')


# # Import modules

# In[2]:


import numpy as np

from analysis_tools import mean_stderr_stats, plot_mean_vs_disorder, plot_mean_arrays_vs_disorder


# # Plot averages versus disorder strength
# 
# ### Parameters to plot

# In[3]:


target_L   = 10
extra_mask = (data_frame['L'] == target_L) & (data_frame['is_final_tau'] == True)


# ## $|[H, \tau ]|^2$ vs $W$

# In[4]:


get_ipython().magic(u'matplotlib notebook')

plot_mean_vs_disorder(data_frame, 'com_norm', '$|[H, \\tau]|^2$', yaxis_scale='log', extra_mask=extra_mask)


# ## $|\tau^2 - I|^2$  vs $W$

# In[5]:


get_ipython().magic(u'matplotlib notebook')

plot_mean_vs_disorder(data_frame, 'binarity', '$|\\tau^2 - I|^2$', yaxis_scale='log', extra_mask=extra_mask)


# # $|\tau|$ vs $W$

# In[7]:


get_ipython().magic(u'matplotlib notebook')

plot_mean_vs_disorder(data_frame, 'tau_norm', '$|\\tau|$', yaxis_scale=None, extra_mask=extra_mask)


# # $\xi$ vs $W$

# In[8]:


get_ipython().magic(u'matplotlib notebook')

plot_mean_vs_disorder(data_frame, 'corr_length', '$\\xi$', yaxis_scale=None, extra_mask=extra_mask)


# ## Weights vs $W$

# In[9]:


get_ipython().magic(u'matplotlib notebook')

Ws_to_plot = [1.0, 3.0, 5.0, 10.0]
L          = target_L #data_frame.L.unique()[0]
sites      = np.arange(-L//2, L//2)
plot_mean_arrays_vs_disorder(data_frame, sites, 'site', 'weights', 'weights', Ws=Ws_to_plot, yaxis_scale='log', extra_mask=extra_mask)


# ## Coefficient histogram vs $W$

# In[10]:


get_ipython().magic(u'matplotlib notebook')

Ws_to_plot = [1.0, 3.0, 5.0, 10.0]
log_coeffs = np.arange(16)
plot_mean_arrays_vs_disorder(data_frame, log_coeffs, '$-\\log_{10}(|$ coefficient $|)$', 'coeff_hist', 'Frequency', Ws=Ws_to_plot, extra_mask=extra_mask)


# In[ ]:




