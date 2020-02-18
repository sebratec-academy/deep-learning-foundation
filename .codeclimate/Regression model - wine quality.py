#!/usr/bin/env python
# coding: utf-8

# **Regression**
# 
# 

# In[2]:


import numpy as np

from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam


# **Introducing new Libraries**
# 
# * **pandas**: Library offering data structures and operation for data manipulation and analysis.
# 
# * **sklearn**: Library providing machine learning algorithms and tools.

# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle


# **The Data**
# 
# We are going to be using a dataset about wine quality. The input variables will give us information about the wine like pH and alcohol percentage. The output variable correlates these inputs to the wine quality.
# 
# This time, the data is not well organized and is inside a csv file. We are going to read this file and prepare the data for training.

# In[4]:


# Loading the dataset file
dataset = pd.read_csv('Datasets/winequalityN.csv')

# Let's take a look at the columns we have in this dataset.
print("Dataset Columns: {}\n".format(dataset.columns.values))


# **Removing invalid rows**
# 
# Sometimes, the datasets might be incomplete. This is the case of this dataset and we will remove rows with missing values.

# In[5]:


# Removes rows with invalid values
n_rows_b4 = dataset.shape[0]
dataset = dataset.dropna(how='any',axis=0)
n_rows_c = dataset.shape[0]
print("{} rows containing invalid data removed.".format((n_rows_b4-n_rows_c)))


# **Shuffling the dataset**
# 
# One way to improve the training performance, is to shuffle the dataset before training.
# 
# When the model learns, it overseers patterns and tries to correlate those patterns to the output. If the occurence of these patterns are not evenly distributed throughout the dataset, the network might focus only on the patterns that occurs the most. Also, we might have missing inputs or outputs values missing when we separate it into train and test data if the dataset if completely sequentially organized.

# In[6]:


# Shuffles the dataset
dataset = shuffle(dataset)


# **Separating into input data and output data**
# 
# Right now, we have the entire dataset inside the same array. We need to separate it so we can tell our model what are the inputs and outputs. 

# In[7]:


# Separate into input and outputs of the network
predictors = dataset.iloc[:,0:12].values
wine_quality = dataset.iloc[:,12].values.astype(np.float32)


# **Categorical Encoding**
# 
# The first column of our dataset is "type". It can be either "white" or "red". The network doesn't understand that. It only understands numbers.
# 
# We could attribute each type with a number. Let's red is 0, and white is 1. But that would not work! The network needs a clearer distinction of what is what.
# 
# Instead, we are going to give each wine time its own neuron. If it is "red", only the first input neuron will be 1. If it "white", only the second neuron will be 1.
# 
# **TODO: The teacher needs to explain the importance of categorical encoding.**

# In[8]:


# Encodes categorized values
print("Cathegory before encoding (10 first): {}".format(predictors[0:10,0]))

ct = ColumnTransformer([("type", OneHotEncoder(),[0])], remainder="passthrough")
predictors = ct.fit_transform(predictors)
print("Cathegory after encoding (10 first):\n{}".format(predictors[0:10,0:2]))


# **Separating into train and test datasets**
# 
# _We are almost there!_
# 
# We removed invalid data, shuffled, and encoded the dataset.
# Now we can finally separate our data to train and test.

# In[20]:


train_ratio = 0.8
train_index = int(train_ratio*predictors.shape[0])
print("Total: {0}. Train: {1}. Test: {2}".format(predictors.shape[0], train_index, predictors.shape[0]-train_index))

x_train = predictors[0:train_index]
y_train = wine_quality[0:train_index]

x_test = predictors[train_index:predictors.shape[0]]
y_test = wine_quality[train_index:predictors.shape[0]]


# In[21]:


print("Input training data shape:", x_train.shape)
print("Example input training data:",x_train[0])
print("Output training data shape:", y_train.shape)
print("Example output training data:",y_train[0])

print("\nInput test data shape:", x_test.shape)
print("Output test data shape:", y_test.shape)


# **Building a regression model**
# 
# It's your time to build a model, train, test, and save it.

# In[38]:


def build_model():
    # Input layer
    inputs = Input([13, ], name='data')
    # Hidden layers
    model = Dense(32, activation='relu')(inputs)
    model = Dense(48, activation='relu')(model)
    model = Dense(48, activation='relu')(model)
    model = Dense(32, activation='relu')(model)
    # Output layer
    outputs = Dense(1, activation='linear', name='fc3')(model)

    # Define the model
    model = Model(inputs=[inputs], outputs=outputs)

    model.compile(optimizer=Adam(lr=2e-5), #Gradient Descend Algorithm.
                  loss='mse',       #MSE = Mean Squared Error
                  metrics=['mae'])  #MAE = Mean Absolute Error
    return model

net = build_model()
print(net.summary())


# In[39]:


net.fit(x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=1000,
        batch_size=128)


# In[40]:


print("Inputs for the Neural Network:", x_test[0])
prediction = net.predict(x_test[2].reshape([1,13]))
print("Predicted value from the neural network:", prediction)
print("Real value:",y_test[0])
print("Error: ",(prediction[0][0] - y_test[0]))


# In[41]:


test_mse_score, test_mae_score = net.evaluate(x_test, y_test)
print("Mean absolute error:",test_mae_score)
print("Mean squared error:",test_mse_score)

predictions = net.predict(x_test).reshape(y_test.shape)
errors = np.abs(predictions - y_test)

print("Min error:",errors.min())
print("Max error:",errors.max())
print("Standard Deviation:",errors.std())


# In[52]:


#50 best
print(np.sort(errors)[0:50])
#50 worst
print(np.sort(errors)[::-1][0:50])


# In[ ]:




