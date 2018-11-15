import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
import keras
import gensim

# Version with timedistributed


def generate(model, start= 1, length_generate = 7) :
    print(reversed_dictionary[start])
    start_word = start
    built_phrase = [start_word]

    seed_text = np.array([start_word])
    seed_text = pad_sequences([seed_text],maxlen=7, padding='post')
    predictions = model.predict_classes(seed_text, verbose=0)


    for i in range(length_generate): 
      predict = predictions[0][i]
      built_phrase.append(predict)
      seed_text = pad_sequences([built_phrase], maxlen=7, padding="post")
      predictions = model.predict_classes(seed_text, verbose =0)

    return built_phrase

def convert_to_integer(array): 
  return [int(b) for b in array]

def get_emotion_timesteps(sequence,emotion_dict,emotion_size=5):

  toReturn = [np.zeros(emotion_size, dtype=bool)]
  
  for i in range(len(sequence)):
    
    word = reversed_dictionary[sequence[i]]
    emotion_vector = emotion_dict[word]
    added_vector = toReturn[i]|emotion_vector
    toReturn.append(added_vector)
  #This code transforms and array of booleans into 0 and on and 1  
  toReturns = [convert_to_integer(emotions) for emotions in toReturn[1:]]

  return np.array(toReturns)

  
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary+1
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
        self.emotion_size = 5

    def generate(self):
        X_emotion= np.zeros((self.batch_size, self.num_steps, self.emotion_size ))
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                data_tmp = self.data[self.current_idx:self.current_idx + self.num_steps]
                x[i, :] = data_tmp
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                
                # get emotion vector
                X_emotion[i] = get_emotion_timesteps(data_tmp,emotion_dict,self.emotion_size)
                
                self.current_idx += self.skip_step
            yield [x,X_emotion], y

df = pd.read_csv('LIWC2015 Results (LICW.csv).csv')
df_emotions = df[['sad','anger','anx','negemo', 'posemo']]
emotions = df_emotions.apply(lambda d : (d!=0.0),axis=1)
df = pd.concat([df[['B']],emotions], axis = 1)

emotion_dict = {}
for w in df.values: 
  emotion_dict[w[0]] = w[1:] 

emotion_dict['nan'] = np.array([False, False, False, False, False])

with open('../Data/cornell movie-dialogs corpus/movie_lines.txt',encoding='utf-8', errors ='ignore') as file:
    data = file.readlines() 

data_array = []
for line in data :
    split_string = line.split('+++$+++')
    dict_values = {'movieID':split_string[2], 'character name':split_string[3], 'utterance': split_string[4]}
    
    #data_array.append(dict_values)
    data_array.append(dict_values['utterance'][1:-1]+' eos ')

def retrieve_data(data_array):
    for utterance in data_array:
        #apply some tokenization of each utterance
        yield gensim.utils.simple_preprocess(utterance, min_len=1)
    
utterances = list(retrieve_data(data_array))

import itertools

d = list(itertools.chain.from_iterable(utterances))
test_d = ' '.join(d)
test_d = test_d.split(' ')
merge_d = [w.lower() for w in test_d]

from collections import Counter

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

print('null' in tokens)
print('nan' in tokens)
print('null' in emotion_dict)
print('nan' in emotion_dict)
print(tokens - set(emotion_dict.keys()))

import sklearn
from sklearn.model_selection import train_test_split
ratio_train_test = 0.25
ratio_test_valid = 0.05

train, test_tmp = sklearn.model_selection.train_test_split(split_data,test_size = ratio_train_test)
test, valid = sklearn.model_selection.train_test_split(test_tmp, test_size = ratio_test_valid)

def data_2_id(dataset, word_to_id):
    data = ''.join(dataset)
    
    data = data.split(' ')
    
    #take care of strings which are empty
    for i,v in enumerate(data):
        if(v==''):
            data.pop(i)
    
    return [word_to_id[w] for w in data]

def new_sequence(sentence):
    newString = sentence +' eos '
    return newString

def final_dataset(dataset):
    data =  [new_sequence(sentence) for sentence in dataset]
    
    return data
    
train_final = final_dataset(train)
train_data = data_2_id(train_final, word2id)

data_path = './'
num_steps = 20
batch_size = 20
skip_step = num_steps+1
hidden_size = 50
num_epochs = 100
input_shape = num_steps
vocabulary = vocab_size




def perplexity(y_true, y_pred):
    cross_entropy = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_pred))
    perplexity = keras.backend.exp(cross_entropy)
    return perplexity

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step)

from keras.utils import plot_model
import graphviz
import pydot

beta = 3
#create the emotional part model
emotion_input = Input(shape=(5,), name='et-1')
g = Dense(100,activation ='sigmoid')(emotion_input)
V = Dense(vocabulary+1)(g)
V_x_beta = Lambda(lambda x: x * beta)(V)
model = Model(inputs=emotion_input, outputs=V_x_beta)

optional_input = Input(shape=(num_steps, 5))
et = TimeDistributed(model)(optional_input)

#CuDNNLSTM

main_input = Input(shape=(input_shape,), dtype='int32', name='ct-1')
embedding = Embedding(input_dim=vocabulary+1, output_dim=200, input_length= input_shape)(main_input)
lstm_layer_1 = LSTM(200,return_sequences=True)(embedding)
lstm_layer_2 = LSTM(200,return_sequences=True)(lstm_layer_1)
ct = TimeDistributed(Dense(vocabulary+1))(lstm_layer_2)

ct_1_plus_et_1 = keras.layers.Add()([ct, et])
softmax = keras.layers.Activation('softmax')(ct_1_plus_et_1)

model = keras.models.Model(inputs=[main_input, optional_input], outputs=[softmax])

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=[perplexity, 'accuracy'])


plot_model(model, to_file='version1.png')
model.summary()





model.fit_generator(train_data_generator.generate(),len(train_data)//(batch_size* num_steps),epochs=num_epochs,callbacks=[checkpointer])



data_path = './'
num_steps = 20
batch_size = 20
skip_step = num_steps+1
hidden_size = 50
num_epochs = 100
input_shape = num_steps
vocabulary = vocab_size




def perplexity(y_true, y_pred):
    cross_entropy = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_pred))
    perplexity = keras.backend.exp(cross_entropy)
    return perplexity

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step)

from keras.utils import plot_model
import graphviz
import pydot

beta = 3
#create the emotional part model
emotion_input = Input(shape=(5,), name='et-1')
g = Dense(100,activation ='sigmoid')(emotion_input)
V = Dense(vocabulary+1)(g)
V_x_beta = Lambda(lambda x: x * beta)(V)
model = Model(inputs=emotion_input, outputs=V_x_beta)

optional_input = Input(shape=(num_steps, 5))
et = TimeDistributed(model)(optional_input)

#CuDNNLSTM

main_input = Input(shape=(input_shape,), dtype='int32', name='ct-1')
embedding = Embedding(input_dim=vocabulary+1, output_dim=200, input_length= input_shape)(main_input)
lstm_layer_1 = LSTM(200,return_sequences=True)(embedding)
lstm_layer_2 = LSTM(200,return_sequences=True)(lstm_layer_1)
ct = TimeDistributed(Dense(vocabulary+1))(lstm_layer_2)

ct_1_plus_et_1 = keras.layers.Add()([ct, et])
softmax = keras.layers.Activation('softmax')(ct_1_plus_et_1)

model = keras.models.Model(inputs=[main_input, optional_input], outputs=[softmax])

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=[perplexity, 'accuracy'])


plot_model(model, to_file='version1.png')
model.summary()





model.fit_generator(train_data_generator.generate(),len(train_data)//(batch_size* num_steps),epochs=num_epochs,callbacks=[checkpointer])




def perplexity(y_true, y_pred):
    cross_entropy = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_pred))
    perplexity = keras.backend.exp(cross_entropy)
    return perplexity

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step)

from keras.utils import plot_model
import graphviz
import pydot

beta = 3
#create the emotional part model
emotion_input = Input(shape=(5,), name='et-1')
g = Dense(100,activation ='sigmoid')(emotion_input)
V = Dense(vocabulary+1)(g)
V_x_beta = Lambda(lambda x: x * beta)(V)
model = Model(inputs=emotion_input, outputs=V_x_beta)

optional_input = Input(shape=(num_steps, 5))
et = TimeDistributed(model)(optional_input)



main_input = Input(shape=(input_shape,), dtype='int32', name='ct-1')
embedding = Embedding(input_dim=vocabulary+1, output_dim=200, input_length= input_shape)(main_input)
lstm_layer_1 = CuDNNLSTM(200,return_sequences=True)(embedding)
lstm_layer_2 = CuDNNLSTM(200,return_sequences=True)(lstm_layer_1)
ct = TimeDistributed(Dense(vocabulary+1))(lstm_layer_2)

ct_1_plus_et_1 = keras.layers.Add()([ct, et])
softmax = keras.layers.Activation('softmax')(ct_1_plus_et_1)

model = keras.models.Model(inputs=[main_input, optional_input], outputs=[softmax])

model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=[perplexity, 'accuracy'])


plot_model(model, to_file='version1.png')
model.summary()



model.fit_generator(train_data_generator.generate(),len(train_data)//(batch_size* num_steps),epochs=num_epochs,callbacks=[checkpointer])

