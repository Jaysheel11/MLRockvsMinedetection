#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the dependencies


# In[16]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[17]:


#Data collection and Data processing


# In[18]:


#loading the data set to pandas dataframe
sonar_data = pd.read_csv('Copy of sonar data.csv',header =None)


# In[19]:


#displays first five rows of our file
sonar_data.head()


# In[20]:


#number of rows and columns
sonar_data.shape


# In[21]:


sonar_data.describe() #describe gives statistical measiures of the data


# In[24]:


sonar_data[60].value_counts() #M--represents mine R--represents rock


# In[25]:


sonar_data.groupby(60).mean()


# In[26]:


#seprating data and labels
X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]


# In[29]:


print(X)
print(Y)


# In[38]:


#Training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)


# In[42]:


print(X.shape,X_train.shape,X_test.shape)
print(X_train)
print(Y_train)


# In[43]:


#model training ---Logistic regression
model = LogisticRegression()


# In[45]:


#training the logistic regression model with training data
model.fit(X_train,Y_train)


# In[46]:


#ModelEvaluation


# In[47]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[48]:


print('Accuracy on training data',training_data_accuracy)


# In[50]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[51]:


print('Accuracy on test data',test_data_accuracy)


# In[52]:


#MAKING A PREDICTIVE SYSTEM


# In[58]:


input_data=(0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#RESHAPING THE NP ARRAY AS WE ARE PREDICTING FOR ONE INSTANCE
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)


# In[59]:


print(prediction)


# In[60]:


if(prediction[0]=='R'):
    print('the object is rock')
else:
    print('it is a mine')


# In[ ]:




