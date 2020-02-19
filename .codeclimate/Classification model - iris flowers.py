#!/usr/bin/env python
# coding: utf-8

# **Classifier**
# 
# Classification involves predicting what class something belongs to. Classifiers can be binary or multi-class, meaning that they either classify something into a binary decision ( yes/no, spam/not spam, hot/cold ) or into several different categories ( blue, yellow, red or green? ). Classification models are a very common use case in deep learning, and they can be used to solve a lot of different problems. 
# 
# Now, we are going to build a classification model that can help us to classify some flower photographs into three different species.

# In[2]:


import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils


# **The Data**
# 
# The Iris flower dataset or Fisher's Iris dataset is a multivariate dataset that quantifies the morphologic variation of Iris flowers of three related species.
# 
# 1. Sepal length in cm
# 2. Sepal width in cm
# 3. Petal length in cm
# 4. Petal width in cm
# 5. Classes:
#   - Iris Setosa
#   - Iris Versicolour
#   - Iris Virginica
# 
#   ![Iris](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png)
# 

# In[6]:


# Download the data
get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O Datasets/iris-data.csv')


# **Loading a csv file**
# 
# This data is organized inside a csv file. We are going to read the values from this file and organize them into arrays.

# In[7]:


dataset = pd.read_csv("Datasets/iris-data.csv", header=None).values
print("The dataset examples:")
print(dataset[0])
print(dataset[50])
print(dataset[100])


# **Separating the inputs and outputs**
# 
# This dataset need to be organized and preprocessed before it can be used inside the neural network. Let's start by separating the input values to an array (X) and the output values to another array (Y).

# In[8]:


# Getting the input data
X = dataset[:,0:4].astype(float)
# Getting the output
Y = dataset[:,4]

print("The dataset examples:")
print("Input:",X[0],"Output:",Y[0])
print("Input:",X[50],"Output:",Y[50])
print("Input:",X[100],"Output:",Y[100])


# **Encoding the labels**
# 
# Remember that neural networks are mathematical functions but our labels are names. We also need to convert the way we represent those names to a numerical form.
# 
# Since we have three species of flowers, so each one of them will be represented by three numbers. Each one of these numbers will also represent an output of the neural network, so we are going to have three outputs. This means that we want only one of the outputs neurons to activate for each class we have.

# In[9]:


# Change the classes to numerical values
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y = np_utils.to_categorical(encoded_Y)

print("The dataset examples:")
print("Iris-setosa")
print("Input:",X[0],"Output:",Y[0])
print("Iris-versicolor")
print("Input:",X[50],"Output:",Y[50])
print("Iris-virginica")
print("Input:",X[100],"Output:",Y[100])


# **Separating the train and test data**
# 
# The network need to be able to perform with data that it hasn't seen during the training process. So we separate a small portion of the dataset out of the training process to better evaluate its performance later.

# In[10]:


# Shuffle the dataset
X, Y = shuffle(X, Y, random_state=0)

# 80% for training and 20% for test
p = int(len(dataset)* 0.8 )

x_train = X[:p]
x_test = X[p:]

y_train = Y[:p]
y_test = Y[p:]


# **Building a regression model**
# 
# * Input layer: 4 neurons (we have 4 inputs)
# * Hidden 2 hidden layers of 6 neurons each.
# * Output layer: 3 neuron (we have 3 classes we are trying to classify).
# * Activation functions: Here we are going to introduce the softmax activation function.

# In[11]:


def build_model():
  # Input layer
  inputs = Input([4, ], name='data')
  # Hidden layers
  model = Dense(6, activation='relu', name='fc1')(inputs)
  model = Dense(6, activation='relu', name='fc2')(model)
  # Output layer
  outputs = Dense(3, activation='softmax', name='fc3')(model)

  # Define the model
  model = Model(inputs=[inputs], outputs=outputs)

  model.compile(optimizer='adam',#Adam(lr=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

net = build_model()
print(net.summary())


# **Training the model**
# 
# Now we have our data separated into inputs and outputs, and training data and testing data. We also have defined our architecture. All that is left to do is fit the data inside our model (train).

# In[15]:


net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=250,
        batch_size=8)


# **Performing Inference**
# 
# Let's infer one element of our test dataset and see the probabilities of the network's prediction.

# In[16]:


prediction = net.predict(x_test[0].reshape([1,4]))
print("Prediction probabilities:")
for i, category in enumerate(encoder.classes_):
  print(category,": %.2f%%" % (prediction[0][i]*100))

print("Real class: ", encoder.classes_[np.argmax(y_test[0])]) 


# **Evaluating the trained neural network**
# 
# A better way to evaluate a classification model is by using a confusion matrix. In this matrix we correlate the predicted class with the desired class.

# In[17]:


print('Confusion Matrix')
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(net.predict(x_test), axis=1)))
print('\nClassification Report')
print(classification_report(np.argmax(y_test, axis=1), np.argmax(net.predict(x_test), axis=1), target_names=encoder.classes_))


# In[ ]:




