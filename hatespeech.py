"""hatespeech.ipynb

Original file is located at
    https://colab.research.google.com/drive/1FVTAKsc3q5ymE8XC9l_Z-AOHA7O76xCq
"""

## Process flow ##

## 1. CSV Comments
## Kaggle data set along with my own comments
## 2. Inputs -> String data type
## 3. labels, and grading of toxicity as tags- would be multi binary
## 4. Pre-processing -> tokenization of labels via help of text vectorization function, and a corresponding vector of every element in the array, check book for better understanding
## 5. Deep Neural Network -> working with sequences here
## 6. Gradio App -> user interface of the DLM model, for testing

from google.colab import drive
drive.mount('/content/drive')

## Steps to do:

## 0. Dependencies and data
## 1. Preprocessing
## 2. Create sequential model
## 3. Make predictions
## 4. Evaluate the model
## 5. Test and Gradio

#adding dependencies

import os
import numpy as np  #
import tensorflow as tf   #dl framework
import pandas as pd   #for reading tabular data
import matplotlib as mb
import sklearn as skl

np.expand_dims

"""# **0. Importing data**"""

#importing data

df = pd.read_csv('/content/drive/MyDrive/train.csv')

df.head()

df.head(8)

df.iloc[0]['comment_text']

df[df.columns[2:]].iloc[3]

df[df.columns[2:]].iloc[6]

#this is imp

df[df['toxic']==1].head()

"""# **1. Pre-processing**"""

# 1.Preprocessing

#for tokenization
from tf_keras.layers import TextVectorization

!pip list

x = df['comment_text']
y = df[df.columns[2:]].values

df.columns

df[df.columns[2:]].values
#basically we created array of it acc to the layout

x

y

MAX_WORDS = 200000   # number of words in vocabulary

# initialize text vectorization

vectorizer = TextVectorization(max_tokens = MAX_WORDS, output_sequence_length=1800, output_mode='int')

type(x)

type(x.values)

vectorizer.adapt(x.values)

#eg that how it vectorizes any text
vectorizer('This is sample text')[:4]

vectorizer.get_vocabulary()

#in this step we will vectorize all the comments_txt i.e. x.values
vectorized_text = vectorizer(x.values)

len(x)

#whats inside vectorized_text, all samples taken
print(vectorized_text)

# creating a tensorflow data pipeline for training
# import tensorflow as tf
## map, cache, batch, prefetech from tenser_slices

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text,y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottle neck

batch_x, batch_y = dataset.as_numpy_iterator().next()

(int(len(dataset)*.7))

# training validation and test partitions
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

train_generator = train.as_numpy_iterator()

train_generator.next()  # going to next pass

train_generator.next()

"""# **2. Creating Sequential Model**"""

# sequental layers
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

model = Sequential()

# Creating the embedding layer
model.add(Embedding(MAX_WORDS+1, 32))

# Creating a LSTM layer, using tanh because of GPU accelaration required by LSTM to run is provided by this 'tanh' activation function
model.add(Bidirectional(LSTM(32, activation='tanh')))
# and we are using bidirectional because it is sentence based and words previously and after targeted word might affect or changw the meaning

# Feature extractor Fully Connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# final layer
model.add(Dense(6, activation='sigmoid')) # so that we can get output between 0  and 1, so that out features can work accordingly

model.compile(loss='BinaryCrossentropy', optimizer='Adam')

model.summary()

# training my deep neural network

history = model.fit(train, epochs=1, validation_data=val)

