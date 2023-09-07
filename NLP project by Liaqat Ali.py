#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NLP project of the module


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import string, os
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[3]:


# reading dataset
df = pd.read_csv("/content/drive/MyDrive/NLP projet/topical_chat.csv")
df.head()


# In[4]:


# basic preprocessing
def process(text):
    text = text.lower().replace('\n', ' ').replace('-', ' ').replace(':', ' ').replace(',', '') \
          .replace('"', ' ').replace(".", " ").replace("!", " ").replace("?", " ").replace(";", " ").replace(":", " ")

    text = "".join(v for v in text if v not in string.punctuation).lower()
    #text = text.encode("utf8").decode("ascii",'ignore')

    text = " ".join(text.split())
    #text+="<eos>"
    return text


# In[5]:


df.message = df.message.apply(process)


# In[6]:


df.head()


# In[7]:


# Vectorize the data.
input_texts = []
target_texts = []
input_words_set = set()
target_words_set = set()

for conversation_index in tqdm(range(df.shape[0])):

    if conversation_index == 0:
        continue

    input_text = df.iloc[conversation_index - 1]
    target_text = df.iloc[conversation_index]

    if input_text.conversation_id == target_text.conversation_id:

        input_text = input_text.message
        target_text = target_text.message

        if len(input_text.split()) > 2 and \
            len(target_text.split()) > 0 and \
            len(input_text.split()) < 30 and \
            len(target_text.split()) < 10 and \
            input_text and \
            target_text:

            target_text = "bos " + target_text + " eos"

            input_texts.append(input_text)
            target_texts.append(target_text)

            for word in input_text.split():
                if word not in input_words_set:
                    input_words_set.add(word)
            for word in target_text.split():
                if word not in target_words_set:
                    target_words_set.add(word)


# In[8]:


input_texts


# In[9]:


target_texts


# In[10]:


import tensorflow_datasets as tfds

import os
import re
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
MAX_SENTENCE_LENGTH = 60


# In[11]:


import tensorflow_datasets as tfds

# Your data preparation code here
# ...

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    input_texts + target_texts, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]


# In[12]:


def tokenize_and_filter(input_texts, target_texts):
  tokenized_inputs, tokenized_outputs = [], []

  for (sentence1, sentence2) in zip(input_texts, target_texts):
    # tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    # check tokenized sentence max length
    if len(sentence1) <= MAX_SENTENCE_LENGTH and len(sentence2) <= MAX_SENTENCE_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)

  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_SENTENCE_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_SENTENCE_LENGTH, padding='post')

  return tokenized_inputs, tokenized_outputs


input_texts, target_texts = tokenize_and_filter(input_texts, target_texts)


# In[14]:


BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': input_texts,
        'dec_inputs': target_texts[:, :-1]
    },
    {
        'outputs': target_texts[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:




