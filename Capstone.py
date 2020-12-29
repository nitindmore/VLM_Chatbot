#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')



with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            #print("with tokenize:\n", wrds)
            wrds = [wrd for wrd in wrds if wrd.lower() not in stopwords]
            #print("without stopwords:\n", wrds)
            wrds = [wrd.lower() for wrd in wrds if wrd not in string.punctuation]
            #print("without punctuation:\n", wrds)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#try:
#model.load('./model.tflearn')
#except:
model.fit(training, output, n_epoch=200, batch_size=8, show_metric=True)
model.save('./model.tflearn')


# In[2]:


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(results)
        #print(tag)

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
            
        print("Bot: ",random.choice(responses))

chat()


# In[3]:


labels


# In[40]:


stopwords


# In[4]:


words


# In[5]:


docs_x


# In[6]:


docs_y


# In[7]:


training[0]


# In[8]:


output[0]


# In[ ]:





# In[ ]:




