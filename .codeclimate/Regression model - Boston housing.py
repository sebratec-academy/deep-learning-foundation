#!/usr/bin/env python
# coding: utf-8

# **Regression**
# 
# Regression models are typically used to predict one value (or a set of values) based on input data. Let's say for example: Predict the price of a car based on the year, fuel consumption, type (sports, compact, SUV), motor power. Or predict the number of sales of a specific product based on month of the year, product price, local economy situation. 
# 
# This is a supervised learning statistical model that correlates the influence of independent variables on dependent variables through fitting a mathematical function according to the behavior of the training data. 

# **Used Python Libraries**
# 
# Let's start by importing the python libraries we are going to be using.
# 
# * **numpy**: Adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.
# 
# * **keras**: Neural network library and abstraction. Can run on top of TensorFlow (Google's neural network library). Designed to enable fast experimentation with deep neural networks.

# In[2]:


import numpy as np

from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.datasets import boston_housing


# **The Data**
# 
# Keras library provides us with a few dataset examples that we can use to experiment with neural networks. The data is already correctly shaped and separated into training data and test data.
# 
# We are going to be using the Boston Housing dataset for this exercise. It correlates the median price (in $1000) of Boston Houses to 13 other parameters (more details in: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
# 
# When training a deep learning model, the data is usually separated into training and test data. The training data is the one used to adjust the neuron's weights, meaning that it is the data the model actually learns from. The test data is used to evaluate the performance of the network during training.

# In[3]:


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print("Input training data shape:", x_train.shape)
print("Example input training data:",x_train[0])
print("Output training data shape:", y_train.shape)
print("Example output training data:",y_train[0])

print("\nInput test data shape:", x_test.shape)
print("Output test data shape:", y_test.shape)


# **Building a regression model**
# 
# Before training, we need to specify our model's architecture. [NN-SVG](http://alexlenail.me/NN-SVG/index.html) is a nice tool to visualize the architecture.
# 
# * Input layer: 13 neurons (we have 13 inputs)
# * Hidden layers: It's up to us to decide. There is not definitive answer. Experimentation is part of developing a deep learning model. But let's start with 2 hidden layers of 32 neurons each.
# * Output layer: 1 neuron (we have 1 output).
# * Activation functions: Here we are going to introduce 2 activation functions: ReLU and Linear (or identity).

# In[4]:


def build_model():
  # Input layer
  inputs = Input([13, ], name='data')
  # Hidden layers
  model = Dense(32, activation='relu', name='fc1')(inputs)
  model = Dense(32, activation='relu', name='fc2')(model)
  # Output layer
  outputs = Dense(1, activation='linear', name='fc3')(model)

  # Define the model
  model = Model(inputs=[inputs], outputs=outputs)

  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])  #MAE = Mean Absolute Error
  return model

net = build_model()
print(net.summary())


# **Training the model**
# 
# Now we have an untrained model. We need to feed it with data and fit the weights around it. We also need to specify another 2 parameters:
# 
# * Epochs: The number of times the network is going to look through the entire dataset and learn.
# * Batch size: The number of examples used for a single learning session. When increasing this number, the training goes faster but also requires more memory. 

# In[5]:


net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=200,
        batch_size=32)


# **Performing Inference**
# 
# After we have a trained model, we can feed the network with new input values and let it predict a new output. In our case, we are predicting the value of a house.
# 
# Let's infer one set of values for a Boston House and check the error.

# In[6]:


print("Inputs for the Neural Network:", x_test[0])
prediction = net.predict(x_test[2].reshape([1,13]))
print("Predicted value from the neural network:", prediction)
print("Real value:",y_test[0])
print("Error: ",(prediction[0][0] - y_test[0]))


# **Evaluating the trained neural network**
# 
# It's always important to evaluate the network and check if it's behavior is acceptable for it's destined application.
# 
# Here we are going to calculate the minimum, maximum, average absolute and standard deviation of the error of our entire test dataset. This doesn't guarantee that future inferences will always stay within this margin of error. It's possible that your network didn't. Meaning there was a set of input values very different from any inside the training dataset, therefore the network wasn't trained for that situation. 

# In[7]:


test_mse_score, test_mae_score = net.evaluate(x_test, y_test)
print("Mean absolute error:",test_mae_score)
print("Mean squared error:",test_mse_score)

predictions = net.predict(x_test).reshape(y_test.shape)
errors = np.abs(predictions - y_test)

print("Min error:",errors.min())
print("Max error:",errors.max())
print("Standard Deviation:",errors.std())


# **Saving your trained neural network**
# 
# Training can take a long time to finish and occupy lots of storage with the data. We don't need to go through that again when we actually need to use the network. Let's save the network weights so we can use it whenever we want.  

# In[8]:


net.save('./regresion_boston.h5')


# **Loading a pretrained neural network**
# 
# We can just load the network and run inferences now that we have the trained model.

# In[9]:


loaded_net = load_model('./regresion_boston.h5')
prediction_loaded = loaded_net.predict(x_test[0].reshape([1,13]))
print("Predicted value from the neural network:",prediction_loaded)


# In[ ]:




