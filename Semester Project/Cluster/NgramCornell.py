import collections
import os
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import gensim
import gensim
import itertools
import sklearn
import graphviz
import pydot

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, CuDNNLSTM, Input,Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils import plot_model


def new_sequence(sentence):
    newString = sentence +' eos '
    return newString

def final_dataset(dataset):
    data =  [new_sequence(sentence) for sentence in dataset]
    
    return data


def data_2_id(dataset, word_to_id):
    data = ''.join(dataset)
    
    data = data.split(' ')
    
    #take care of strings which are empty
    for i,v in enumerate(data):
        if(v==''):
            data.pop(i)
    
    return [word_to_id[w] for w in data]

def convert_to_integer(array): 
  return [int(b) for b in array]

def get_emotion_timesteps(sequence,emotion_dict,reversed_dictionary, emotion_size=5):

  toReturn = [np.zeros(emotion_size, dtype=bool)]
  
  for i in range(len(sequence)):
    
    word = reversed_dictionary[sequence[i]]
    emotion_vector = emotion_dict[word]
    added_vector = toReturn[i]|emotion_vector
    toReturn.append(added_vector)
  #This code transforms and array of booleans into 0 and on and 1  
  toReturns = [convert_to_integer(emotions) for emotions in toReturn[1:]]

  return toReturns
  
def retrieve_data(data_array):
    for utterance in data_array:
        #apply some tokenization of each utterance
        yield gensim.utils.simple_preprocess(utterance, min_len=1)


class KerasBatchGenerator(object):

    def __init__(self, X,time_steps, reversed_dictionary):
        self.train_data = X
        self.time_steps = time_steps
        #self.aux_input = aux_input
        #self.batch_size = batch_size
        
        self.current_idx = 0 
        self.reversed_dictionary = reversed_dictionary
        

        
    def get_data(self,train_data, time_steps, current_idx):
 
        d = np.array(train_data[current_idx:current_idx +time_steps])

        emotion_vectors = get_emotion_timesteps(d[:-1], emotion_dict, self.reversed_dictionary)

        sequences = []

        for i in range(2,time_steps+1) :
            #[0-i)
             sequences.append(d[:i])

        #proceed to pad the sequences
        seq = pad_sequences(np.array(sequences), maxlen=time_steps, padding='pre')

        X = seq[:,:-1]
        y = seq[:,-1] #get the true y

        # for example the label 2 will have at the 0,1 vector at position[2] equals to 1 and the rest 0's
        # vocab_size + 1 because the vocabulary index starts at 1 and not at 0
        y = to_categorical(y, num_classes=vocab_size+1)
        
        

        return (X,np.array(emotion_vectors),y)
        
    
    def generate(self):
          
        skip_steps = self.time_steps - 1
        while True:
            if(self.current_idx>=len(self.train_data)):
               self.current_idx = 0
            
            

            data, emotion, target = self.get_data(train_data, time_steps, self.current_idx)
            self.current_idx += skip_steps
         
            yield([data,emotion],target)
        


df = pd.read_csv('../Data/LIWC2015 Results (LICW.csv).csv')
df_emotions = df[['sad','anger','anx','negemo', 'posemo']]
emotions = df_emotions.apply(lambda d : (d!=0.0),axis=1)
df = pd.concat([df[['B']],emotions], axis = 1)


emotion_dict = {}
for w in df.values: 
  emotion_dict[w[0]] = w[1:]

emotion_dict['nan'] = np.array([False,False,False,False,False])

with open('../Data/cornell movie-dialogs corpus/movie_lines.txt',encoding='utf-8', errors ='ignore') as file:
    data = file.readlines() 

data_array = []
for line in data :
    split_string = line.split('+++$+++')
    dict_values = {'movieID':split_string[2], 'character name':split_string[3], 'utterance': split_string[4]}
    data_array.append(dict_values['utterance'][1:-1]+' eos ')

utterances = list(retrieve_data(data_array))


d = list(itertools.chain.from_iterable(utterances))
test_d = ' '.join(d)
test_d = test_d.split(' ')
merge_d = [w.lower() for w in test_d]


words = Counter(merge_d)
words = [word for word in test_d if(words[word] > 1)]

counter = Counter(words)
words_for_LIWC = counter.keys()
words_for_LIWC = list(words_for_LIWC)

tokens = set(words)

vocab_size = len(tokens)
word2id = dict(zip(tokens,range(1,vocab_size+1)))
reversed_dictionary = dict(zip(word2id.values(), word2id.keys()))


complete_string = ' '.join(words)
split_string = complete_string.split('eos')
#remove last string which is a space 
split_string = split_string[:-1]

split_data = np.array(split_string)
clean = [sentence[1:-1] for sentence in split_data[1:]]

clean.insert(0, split_data[0][:-1])
split_data = np.array(clean)

ratio_train_test = 0.25
ratio_test_valid = 0.05

train, test_tmp = sklearn.model_selection.train_test_split(split_data,test_size = ratio_train_test)
test, valid = sklearn.model_selection.train_test_split(test_tmp, test_size = ratio_test_valid)

train_final = final_dataset(train)
train_data = data_2_id(train_final, word2id)

time_steps = 20
input_shape = time_steps-1
beta = 0.5

main_input = Input(shape=(input_shape,), dtype='int32', name='ct-1')
embedding = Embedding(input_dim=vocab_size+1, output_dim=200, input_length= input_shape)(main_input)
lstm_layer_1 = LSTM(200,return_sequences=True)(embedding)
lstm_layer_2 = LSTM(200)(lstm_layer_1)
U = Dense(vocab_size+1)(lstm_layer_2)

#TODO add drop out layer

optional_input = Input(shape=(5,), name='et-1')
g = Dense(100,activation ='sigmoid')(optional_input)
V = Dense(vocab_size+1)(g)
V_x_beta = Lambda(lambda x: x * beta)(V)

ct_1_plus_et_1 = keras.layers.Add()([U,V_x_beta])
softmax = keras.layers.Activation('softmax')(ct_1_plus_et_1)
model = keras.models.Model(inputs=[main_input, optional_input], outputs=[softmax])

#TODO remember to add in the perplexity score
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

plot_model(model, to_file='version2.png')

train_data_generator = KerasBatchGenerator(train_data, time_steps, reversed_dictionary)

model.summary()
model.fit_generator(train_data_generator.generate(),len(train_data)//(time_steps),epochs=1)