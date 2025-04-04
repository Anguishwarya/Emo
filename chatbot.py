#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


from bardapi import Bard
import os
import time


# In[3]:


# Replace XXXX with the values you get from __Secure-1PSID
os.environ['_BARD_API_KEY']="0001 0001 0039 0000 00e9 0000 0000 00"


# In[5]:


# Set your input text
# input_text = "Why is the sky blue?"
def get_bard_response(user_input):
    try:
        response = Bard().get_answer(user_input)['content']
        return response
    except Exception as e:
        return "Sorry, I couldn't process your request right now."



# In[ ]:




