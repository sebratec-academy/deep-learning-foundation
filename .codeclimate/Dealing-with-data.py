#!/usr/bin/env python
# coding: utf-8

# # Dealing with data
# 
# One of the most important parts about deep learning is data. Data is pure raw information, human and machine representation of the observation of the world. Everything can be represented in data, including our literature, arts and science. In this lab, we are going to deal with data, apply some techniques to make our data ready to be applied in a machine learning model, and after that, separate our dataset in training, validation and test datasets and feed that data to a deep learning model. 

# ## Acquiring data
# 
# Before you can even prepare your dataset, you need to acquire one. Unless you already work in a place where you already have that data ready for you, you need to collect it yourself. There are a lot of places where you can acquire data, for example:
# 
# - [Datasets subreddit](https://www.reddit.com/r/datasets/), a community where you can share, request and discuss about datasets;
# 
# - [Kaggle](https://www.kaggle.com/), where you can search for a lot of different datasets, jupyter notebooks applying that data, and even engage in competitions;
# 
# - [Awesome datasets in Github](https://github.com/awesomedata/awesome-public-datasets), a curated list of datasets of a lot of different subjects, hosted in Github.
# 
# Sometimes, your data will not be easily available, and you will have to get your hands dirty to collect it. While it is beyond the scope of this lab, it is worth to mention that common data collection techniques involve the usage of sensors and [web-scraping](https://en.wikipedia.org/wiki/Web_scraping).
# 

# ## Preprocessing the data
# 
# Let's say we downloaded the records of a HR department of a company specialized in engineering. This dataset in particular, is about the hires for a software engineering position.

# In[1]:


# First, let's import our packages as usual.
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils


# In[2]:


# Then, let's load our dataset.
import os.path
path = "./Datasets/hires.csv"
if os.path.isfile(path) :
    dataset = pd.read_csv(path)
else:
    dataset = pd.read_csv("Neural-networks/" + path)

# Let's take a look at the columns we have in this dataset.
print("Dataset Columns: {}\n".format(dataset.columns.values))


# As you can see, we have the following features:
# 
# - name: the candidate name;
# - interview_score: a subjective grade that was given to the performance of the candidate in the hiring interview;
# - years_of_experience: how many years this particular candidate has in the software engineering industry
# 
# And a class:
# 
# - hired: which means if the candidate ended up being hired or not;
# 
# 
# Now, let's take a closer look at our dataset: 

# In[3]:


# To see some of the data inside our dataset, simply use the dataset variable.
dataset


# There are some incomplete data inside our dataset. Incomplete data is, most of the times, bad data. So, we need to remove them.

# In[4]:


import math
# Remove empty rows from dataset
for index, data in dataset.iterrows():
    if(math.isnan(data.interview_score) or math.isnan(data.years_of_experience)):
        dataset.drop(index = index, inplace = True)

# Show the updated dataset        
dataset


# We also need to decide what features are important for the problem we are trying to solve. Deciding what features are important, is a key part of machine learning. Remember, bad data equals to bad results, and wrong features are bad data.
# 
# In this case, the interview score and the years of experience of the candidate are important features, but the name of the candidate is completely irrelevant for our model, so we can just take it out from our dataset:

# In[5]:


# Delete the column name
del dataset['name']
# Show our updated dataset
dataset


# Another thing that you should have in mind, is that "True" and "False", doesn't mean anything to a machine. We need to convert that into something that a computer will understand:

# In[6]:


# Convert true to 1, and false to 0
dataset.hired = dataset.hired.astype(int)
dataset


# In[7]:


# Let's plot the dataset, so we can see who got hired and those who didn't.
dataset.plot(kind='scatter', x="interview_score", y="years_of_experience", c="hired", colormap='jet')


# Now, let's separate our dataset into test and training datasets:

# In[8]:


X = dataset[["years_of_experience","interview_score"]].astype(float)
Y = dataset[["hired"]]
Y = np.ravel(Y)

# Change the classes to numerical values
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
Y = np_utils.to_categorical(encoded_Y)

# Separate the dataset into test and training datasets
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=0.25,)


# Now let's build a simple classification neural network and train it:

# In[9]:


def build_model(): 
  # Input layer
  inputs = Input([2, ], name='data')
  # Hidden layers
  model = Dense(128, activation='relu')(inputs)
  # Output layer
  outputs = Dense(2, activation='softmax', name='fc3')(model)

  # Define the model
  model = Model(inputs=[inputs], outputs=outputs)

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

net = build_model()
print(net.summary())


# In[10]:


net.fit(X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=200,
        batch_size=8)


# In[11]:


# Print model performance
test_loss, test_acc = net.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Predict if a new candidate with interview score of 0.6 and 15 years of experience will be hired
input = np.array([[15,0.6]])
prediction = net.predict(input)
for index, predict in enumerate(prediction):
  if predict[0]*100 > predict[1]*100:
    print("Candidate probably will not be hired")
  else:
    print("Candidate probably will be hired")


# In[ ]:




