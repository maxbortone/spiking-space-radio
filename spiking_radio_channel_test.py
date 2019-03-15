#!/usr/bin/env python
# coding: utf-8

# In[5]:


from utils.channel import send
from utils.modulator import AsynchronousDeltaModulator, modulate
from utils.plotting import plot_sample_with_labels


# In[2]:


input_string = "penis"
time_sample, sample, time_labels, labels, bins = send(input_string, m=4)


# In[3]:


settings = {
    'thrup': 0.1,
    'thrdn': 0.1,
    'resampling_factor': 10
}


# In[6]:


modulators = [
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor']),
    AsynchronousDeltaModulator(settings['thrup'], settings['thrdn'], settings['resampling_factor'])
]


# In[7]:


indices, times, time_stim, signal = modulate(modulators[0], modulators[1], time_sample, sample, resampling_factor=settings['resampling_factor'])


# In[9]:


plot_sample_with_labels(signal, time_stim, indices, times, labels, time_labels, bins, figsize=(12, 9))

