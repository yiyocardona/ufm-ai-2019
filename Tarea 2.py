#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Tarea 2

import torch


# In[3]:


#Ej.1

a = torch.Tensor(list(range(9)))


print ("El tamaño de a es de", a.size())
print ("El stride de a es de ", a.stride())
print ("El offset de a es de ", a.storage_offset())


b = a.view(3,3)
print ("El valor de b[1, 1] es de ", b[1, 1])
c = b[1:, 1:]
print ("El tamaño de c es de", c.size())
print ("El stride de c es de ", c.stride())
print ("El offset de c es de ", c.storage_offset())


# In[4]:


#Ej 2
tcos = torch.Tensor([0,1,2])
tcos2 = torch.cos(tcos)
print(tcos2)
#In place
torch.cos_(torch.Tensor([0,1,2]))


# In[5]:


#Ej 3
un = torch.Tensor([[0,1,2],[0,1,2]])
un.unsqueeze_(0).shape


# In[6]:


#Ej 4
un.squeeze_(0).shape


# In[7]:


#Ej 5
al = torch.randint(low = 3, high = 7, size = (5,3))
al


# In[8]:


#Ej 6
dist_normal = torch.randn(3,3)
dist_normal


# In[9]:


#Ej 7
n = torch.Tensor([1,1,1,0,1])
index = torch.nonzero(n)
index


# In[10]:


#Ej 8
al = torch.rand(3,1).t()
tensor_f = torch.cat([al, al, al, al], dim = 0).t()
tensor_f


# In[11]:


#Ej 9
f = torch.randn(3,4,5)
d = torch.rand(3,5,4)
g = torch.matmul(f,d)
g


# In[12]:




#Ej 10
q = torch.randn(3,4,5)
w = torch.rand(5,4)
r = torch.matmul(q,w)
r


# In[ ]:




