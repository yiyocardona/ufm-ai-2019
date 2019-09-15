#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Tarea 4


# In[13]:



import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# In[14]:



vinos = np.loadtxt('winequality-white.csv', delimiter=';', skiprows = 1)
data_x = vinos[:,0:11]
data_y = vinos[:,-1]


# In[15]:



#arrays a tensores

data_x = torch.Tensor(data_x)
data_y = torch.LongTensor(data_y).unsqueeze(1)
num = data_y.shape[0]
data_y = torch.zeros(num,11).scatter_(1,data_y,1)


# In[16]:


#separacion de train y test
n_samples = data_x.shape[1]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
train_x = data_x[train_indices]
train_y = data_y[train_indices]
val_x = data_x[val_indices]
val_y = data_y[val_indices]


# In[18]:



def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x)
        train_loss = loss_fn(train_t_p, train_y)

        with torch.no_grad():
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


            
class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(11, 30)
        self.output_linear = nn.Linear(30, 11)


    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


func_model = SubclassFunctionalModel()
func_model


# In[19]:


#Magic
get_ipython().run_cell_magic('time', '', '#definir el optimizador y realizar el training loop con el modelo secuencial\noptimizer = optim.SGD(func_model.parameters(), lr=1e-3)\n\ntraining_loop(\n    n_epochs=5000,\n    optimizer=optimizer,\n    model=func_model,\n    loss_fn=nn.MSELoss(),\n    train_x = train_x,\n    val_x = val_x,\n    train_y = train_y,\n    val_y = val_y)')


# In[ ]:


#Respuestas
#1 El modelo empleado se tardo 3.2 segundos
#2 La estructura o arquitectura del modelo es lo que mas influye en el tiempo del trining del modelo.
#3 Para tener una mayor perdida de loss podemos aumentar las neuronas para que el perfomance en si del modelo mejore.

#4.Graficar la data en su totalidad es muy complicado y no seria de utilidad, ya que existen tantas variables a documentar que cualquier analisis visual seria de alta dificultad. Se pudiera realizar utilizando una gran cantidad de features en la grafica como por ejemplo graficar en los ejes x,y,z, diferencia de figuras, diferencia de tamaño, diferencia de color.


# In[20]:







# In[ ]:


#definir el training loop
def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x)
        train_loss = loss_fn(train_t_p, train_y)

        with torch.no_grad():
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


# In[ ]:


#definir la subclase del modelo, en este caso va a tener 11 inputs, 22 neuronas, y 11 outputs, ya que se tiene en one hot el resultado
class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(11, 30)
        self.output_linear = nn.Linear(30, 11)


    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


func_model = SubclassFunctionalModel()
func_model


# In[ ]:


get_ipython().run_cell_magic('time', '', '#definir el optimizador y realizar el training loop con el modelo secuencial\noptimizer = optim.SGD(func_model.parameters(), lr=1e-3)\n\ntraining_loop(\n    n_epochs=5000,\n    optimizer=optimizer,\n    model=func_model,\n    loss_fn=nn.MSELoss(),\n    train_x = train_x,\n    val_x = val_x,\n    train_y = train_y,\n    val_y = val_y)')






# In[21]:


#Respuestas

#1.El modelo que se empleo en este notebook tarda alrededor de 2.3 segundos, lo cual es mucho mas que el modelo visto en clase, que se realizaba casi de manera instantanea.

#2.El factor principal que contribuye al tiempo de entrenamiento es la estructura de modelo, lo cual incluye el tipo de modelo i.e. su complejidad y el numero de capas, y  las neuronas en cada capa.

#3.Incrementar el numero de neuronas atribuye a una perdida de loss, lo cual mejora nuestro modelo.

#4.Graficar la data en su totalidad es muy complicado y no seria de utilidad, ya que existen tantas variables a documentar que cualquier analisis visual seria de alta dificultad. Se pudiera realizar utilizando una gran cantidad de features en la grafica como por ejemplo graficar en los ejes x,y,z, diferencia de figuras, diferencia de tamaño, diferencia de color.


# In[ ]:




