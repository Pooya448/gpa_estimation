#!/usr/bin/env python
# coding: utf-8

# ## 1. Import and process data
#
#     - In this section, I import the data from files and normalize it, then I split the data to training and test sets
#
# ### 1.1. Import necessary packages

# In[82]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from keras import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization


# ### 1.2. Import data
#     - Import data from excel files using pandas, change nan values with 0 and convert it to numpy array

# In[83]:


X = pd.read_excel(r'Elearning-Data-cut.xls', sheet_name=0).fillna(0).to_numpy()
y1 = pd.read_excel(r'Elearning-Data-cut.xls', sheet_name=1).fillna(0).to_numpy()
y2 = pd.read_excel(r'Elearning-Data-cut.xls', sheet_name=2, usecols=[0]).fillna(0).to_numpy()


# ### 1.3. Normalize data
#     - Normalizing features using Standard Scalar from sklearn

# In[84]:


sc = StandardScaler()
X = sc.fit_transform(X)


# ### 1.4. Split data
#     - Splitting data in train and test sets using train_test_split method from sklearn

# In[85]:


X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.15)


print(X_train.shape)
print(X_test.shape)

print(y1_train.shape)
print(y1_test.shape)

print(y2_train.shape)
print(y2_test.shape)


# ## 2. Building the binary classifier model and regression model
#
#     - In this section, I build 2 training models using Keras
#
#     - For binary classification I use a 3 layer neural network (2 hidden and 1 output), each containing 200, 100 and 1 neuron respectively.
#
#     - For regression I use a 3 layer neural network (2 hidden and 1 output), each containing 200, 100 and 1 neuron respectively. After each layer's activation, I use a BatchNormalization layer for normalizing outputs.
#
# ### 2.1. Building binary classifier model

# In[86]:


classifier = Sequential()

# Batch Normalization Layer
classifier.add(BatchNormalization())

# First Hidden Layer
classifier.add(Dense(200, activation='relu', kernel_initializer='random_normal'))

# Second  Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))

# Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# using adam optimizer alongside with binary cross entropy loss function
classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['mean_squared_error','accuracy'])


# ### 2.2. Building regression model

# In[87]:


model = Sequential()

# Batch Normalization Layer
model.add(BatchNormalization())

# First Hidden Layer
model.add(Dense(200, activation='relu', kernel_initializer='random_normal'))

# Batch Normalization Layer
model.add(BatchNormalization())

# Second  Hidden Layer
model.add(Dense(100, activation='relu', kernel_initializer='random_normal'))

# Batch Normalization Layer
model.add(BatchNormalization())

# Output Layer
model.add(Dense(1, kernel_initializer='normal'))

# using adam optimizer alongside with mean suared loss function
model.compile(optimizer ='adam',loss='mse', metrics =['mean_squared_error'])


# ## 3. Training and evaluation
#
# ### 3.1. Binary Classification

# In[97]:


progress = classifier.fit(X_train,y2_train, batch_size=32, epochs=1000)


# In[98]:


print('\nEvaluation result:')
eval_model=classifier.evaluate(X_test, y2_test)


# In[99]:


# Calculating the Confusion Matrix

y_pred=classifier.predict(X_test)
y_pred =(y_pred > 0.5)

cm = confusion_matrix(y2_test, y_pred)
print(cm)


# In[100]:


# Plotting error and accuracy

plt.plot(progress.history['mean_squared_error'], label='MSE')
plt.plot(progress.history['accuracy'], label='Accuracy')
plt.title('MSE')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="center right")
plt.show()


# ### 3.2. Regression

# In[101]:


progress = model.fit(X_train,y1_train, batch_size=32, epochs=1000)


# In[102]:


prediction = model.predict(X_test)
print('Test set MSE:')
mse = mean_squared_error(y1_test, prediction)
print(mse)
print('Square root of Test set MSE:')
print(np.sqrt(mse))


# In[104]:


# Plotting error for training set

plt.plot(progress.history['mean_squared_error'], label='MSE')
plt.title('MSE')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.show()
