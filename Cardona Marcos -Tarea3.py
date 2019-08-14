#!/usr/bin/env python
# coding: utf-8

# ## TAREA 3

# ### Ejercicios
# 
# #### Redefinan el model a w2 * t_u ** 2 + w1 * t_u + b
# 
# 
# Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?  
# Que partes se mantuvieron iguales?  
# El loss resultante es mas alto o bajo despues de entrenamiento?  
# El resultado es mejor o peor?
# 
# 
# 1. Se cambio la funcion de model() para que este pudiera aceptar el w2  
# 2. Las funciones quedaron exactamente igual  
# 3. El loss despues del entrenamiento fue mas bajo  
# 4. Error quedo mas alto pero el resultado fue mejor

# In[8]:


import numpy as np
import torch
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] 
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] 
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)




# In[9]:


def model(t_u, w1,w2, b):
     return w2 * t_u ** 2 + w1 * t_u + b


# In[10]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[11]:



def training_loop(model, n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params


# In[31]:


params = torch.tensor([1.0 , 1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)


# In[34]:


training_loop(model,
              n_epochs=3000,
              optimizer=optimizer,
              params = params,
              t_u = t_u, 
              t_c = t_c)


# In[ ]:




