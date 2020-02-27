#!/usr/bin/env python
# coding: utf-8

# # Common problems faced by neural networks
# 
# When working with neural networks, you can face some problems. Sometimes, your model may be really good when predicting your training datasets, but performing poorly when seeing completely different data. Maybe, it just won't perform well at all, will have consistently huge losses or make bad predictions. 
# 
# Sometimes you do not have enough data to train your model, or your data is too disperse, the values are too different from each other, the training process is taking too long, the list goes on.
# 
# We will explore some of these problems, and apply some techniques to fix, and prevent them from happening.

# ## Underfitting
# 
# When your neural network is not performing well in it's own training set, and even worse in data that it has never seen before, is a phenom called underfitting. There are a lot of techniques that we can use to solve underfitting, and usually we have to analyze what is wrong with our model or data, and apply the correct technique to fix it.
# 
# Let's take a look at a neural network that is suffering from underfitting:

# In[1]:


# Import the necesary packages
import numpy as np
import keras
from matplotlib import pyplot as plt
 
from keras.layers import Dense, Input, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.datasets import boston_housing


# In[2]:


# We will use the boston housing dataset in this example. Let's separate the data into training and test datases.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# In[3]:


# Let's build a very simple network, with just one hidden layer with only two neurons inside it
def build_model():
  inputs = Input([13, ], name='data')

  model = Dense(2, activation='relu', name='fc1')(inputs)
  outputs = Dense(1, activation='linear', name='fc3')(model)

  model = Model(inputs=[inputs], outputs=outputs)
  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])
  return model

net = build_model()
print(net.summary())


# In[4]:


# And let's train it using only 50 epochs:
history = net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=32)


# In[5]:


# Ok, now let's calculate the error of the network and plot it:
def plot_loss_and_error():
  test_mse_score, test_mae_score = net.evaluate(x_test, y_test)
  print("Mean absolute error:",test_mae_score)
  print("Mean squared error:",test_mse_score)

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

plot_loss_and_error()


# As we can see, our errors are enourmous. Remember, when analyzing this kind of graphic, the lower the validation loss, the better our model is. And in this case, the error is really big. We must improve our network somehow, or else it will perform poorly when exposed to data it has never seen before.
# 
# In the network exposed above, we have two serious mistakes:
# 
# - The network is very simple. Just a layer with two neurons is not enough, we must add complexity. Some more layers with more neurons should do the trick;
# 
# - We are not training it for long enough. We are only making fifty training iterations, we need to train it for longer.

# In[6]:


# Let's try again. What about a lot of layers and neurons?
def build_model():
  inputs = Input([13, ], name='data')

  model = Dense(512, activation='relu', name='fc1')(inputs)
  model = Dense(512, activation='relu', name='fc2')(model)
  model = Dense(512, activation='relu', name='fc3')(model)
  model = Dense(512, activation='relu', name='fc4')(model)
  model = Dense(512, activation='relu', name='fc5')(model)
  model = Dense(512, activation='relu', name='fc6')(model)
  model = Dense(512, activation='relu', name='fc7')(model)
    
  outputs = Dense(1, activation='linear', name='fc8')(model)

  model = Model(inputs=[inputs], outputs=outputs)
  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])
  return model

net = build_model()
print(net.summary())


# In[7]:


# Let's add a lot of epochs. What can go wrong?
history = net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=500,
        batch_size=32)


# In[8]:


# Let's see how it behaves:
plot_loss_and_error()


# If you take a closer look at the graph above, you are seeing a clear example of overfitting. When you compare your train and validation losses, you should have in mind that:
# 
# - It's normal for there to be a small difference between them.
# - If both metrics are moving in the same direction, everything is fine.
# - If the validation metric begins to stagnate while the training metric continues to improve, you are probably close to overfitting.
# - If the validation metric is going in the wrong direction, the model is clearly overfitting.
# 
# ## Overfitting
# 
# Overfitting is what happens when your model becomes so good at learning it's training dataset that it becomes bad at analyzing data that it has never seen before. In deep learning, fitting your model to your dataset is not a big deal. The real challenge lies in generalizing your model, making it good at predicting data that is outside of it's test dataset.
# 
# The issues with our previous neural network are the following:
# 
# - It is a lot more complex than it should be. We need to redesign it.
# - We need to lower our epochs
# - We are not using any regularization techniques.
# 
# Let's start by fixing the easiest problems:

# In[9]:


# Let's use a more reasonable architecture this time.
def build_model():
  inputs = Input([13, ], name='data')

  model = Dense(32, activation='relu', name='fc1')(inputs)
  model = Dense(32, activation='relu', name='fc2')(model)
    
  outputs = Dense(1, activation='linear', name='fc3')(model)

  model = Model(inputs=[inputs], outputs=outputs)
  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])
  return model

net = build_model()
print(net.summary())


# In[10]:


# Let's also lower our epochs.
history = net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=200,
        batch_size=32)


# In[11]:


# And let's plot the losses graph once more:
plot_loss_and_error()


# Now it looks really better. But there is something you should notice here. Do you see how the losses are not changing much after the epoch 100, give or take? This is a perfect opportunity to apply a technique called **early stopping**.
# 
# ## Early stopping
# 
# Early stopping is basically stopping is basically lowering the amount of epochs you are training your model for. Since the model is not improving much after around epoch 100, maybe we should lower our amount of epochs.

# In[12]:


def build_model():
  inputs = Input([13, ], name='data')

  model = Dense(32, activation='relu', name='fc1')(inputs)
  model = Dense(32, activation='relu', name='fc2')(model)
    
  outputs = Dense(1, activation='linear', name='fc3')(model)

  model = Model(inputs=[inputs], outputs=outputs)
  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])
  return model

net = build_model()
print(net.summary())

# Let's use half of the epochs we did before.
history = net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100, 
        batch_size=32)


# In[13]:


# And analyze the results:
plot_loss_and_error()


# In the example above, applying early stopping to our model not only saved some time and computing power, but also lowered our loss.
# 
# Now, let's get back to our over complex example, apply regularization to it and see what happens:

# ## Regularization
# 
# The main idea behind regularization is to reduce overfitting. It does that by reducing the complexity of neural network during the training process. Among the popular regularization techniques, we have L1 and L2, which penalize complex models by adding a value to the error function when the weights are high.
# By adding a value to the loss function, you force the weights to lower their values. Smaller weights reduce the impact of the hidden neurons, thus making these neurons neglectable and reducing the overall complexity of the neural network.
# 
# The most common among these techniques is L2, which is also called *Ridge Regression* or *Weight decay*. L1 is also called *Lasso Regression*, and the difference between them is penalty term, or the value that it adds to the error function. 
# 
# - L1 regularization encourage the weight values to be zero. 
# - L2 regularization encourages the weights to go towards zero, but not exactly zero.
# 
# We also have the dropout technique, which randomly disables a few neurons inside our network. By doing this, the complexity of the network will be reduced, because some neurons, and even sometimes entire layers will be disabled, thus making it a simpler network.

# In[14]:


# Let's apply L1 and L2 regularization and spice up the network with some dropouts.
def build_model():
  inputs = Input([13, ], name='data')

  # Notice how the regularization can be applied to each layer, instead to the entire network
  model = Dense(512, activation='relu', name='fc1')(inputs)
  model = Dense(512, activation='relu', name='fc2')(model)
  model = Dropout(0.1)(model)
  model = Dense(512, activation='relu', name='fc3')(model)
  model = Dense(512, activation='relu', name='fc4')(model)
  model = Dropout(0.1)(model)
  model = Dense(512, activation='relu', name='fc5')(model)
  model = Dropout(0.1)(model)  
  model = Dense(512, activation='relu', name='fc6',kernel_regularizer=keras.regularizers.l1_l2(l1=0.5, l2=0.01))(model)
  model = Dense(512, activation='relu', name='fc7',kernel_regularizer=keras.regularizers.l1_l2(l1=0.05, l2=0.01))(model)
    
  outputs = Dense(1, activation='linear', name='fc8')(model)

  model = Model(inputs=[inputs], outputs=outputs)
  model.compile(optimizer=Adam(lr=4e-5), #Gradient Descend Algorithm.
                loss='mse',       #MSE = Mean Squared Error
                metrics=['mae'])
  return model

net = build_model()
print(net.summary())

# same epochs as the complex example.
history = net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=500, 
        batch_size=32)


# In[15]:


# And analyze the results:
plot_loss_and_error()


# When you compare this new graph with the graph from the same network, but with no regularization techniques, you will notice how closer the losses over time are closer to each other. So, even though our losses are slightly higher, our model is generalizing really better. This is the real goal when designing neural networks, finding the balance between the accuracy and ability to generalize with data outside the training dataset.

# In[ ]:




