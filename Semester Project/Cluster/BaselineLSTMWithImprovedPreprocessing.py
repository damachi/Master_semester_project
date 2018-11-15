
# coding: utf-8

# In[286]:


import gensim
from gensim.models import Word2Vec
import string
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, CuDNNLSTM
from keras.callbacks import ModelCheckpoint


# In[185]:


def retrieve_data(data_array):
    for utterance in data_array:
        #apply some tokenization of each utterance
        yield gensim.utils.simple_preprocess(utterance, min_len=1)
        
def data_2_id(dataset, word_to_id):
    data = ''.join(dataset)
    
    data = data.split(' ')
    
    #take care of strings which are empty
    for i,v in enumerate(data):
        if(v==''):
            data.pop(i)
    
    return [word_to_id[w] for w in data]

def new_sequence(sentence):
    newString = sentence.append('eos')
    return newString

def final_dataset(dataset):
    data =  [new_sequence(sentence) for sentence in dataset]
    
    return data

def perplexity(y_true, y_pred):
    cross_entropy = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, y_pred))
    perplexity = keras.backend.exp(cross_entropy)
    return perplexity


# In[242]:


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


    def generate(self):

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
                
                self.current_idx += self.skip_step
            yield x, y


# In[96]:


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[13]:


with open('../Data/cornell movie-dialogs corpus/movie_lines.txt',encoding='utf-8', errors ='ignore') as file:
    data = file.readlines() 

data_array = []
for line in data :
    split_string = line.split('+++$+++')
    dict_values = {'movieID':split_string[2], 'character name':split_string[3], 'utterance': split_string[4]}
    
    #data_array.append(dict_values)
    data_array.append(dict_values['utterance'][1:-1]+' eos ')


# In[20]:


for i, phrase in enumerate(data_array):
    data_array[i] = [word.strip(string.punctuation) for word in phrase.split(" ")]


# In[28]:


for i,phrase in enumerate(data_array):
    data_array[i] = list(filter(lambda word : word != '',phrase))


# In[113]:



#join sequences
d = list(itertools.chain.from_iterable(data_array))
lowered_d = [w.lower() for w in d]

#reduce vocabulary by removing infrequent words so we go from 30000 to 10000 perhaps increasing training speed
counter = Counter(lowered_d)
words = [word for word in lowered_d if(counter[word] > 10)]

#remove numbers
words = list(filter(lambda d : not d.isdigit(),words))


# Vocabulary

# In[114]:


tokens = set(words)
print(len(tokens))


# In[115]:


words_for_LIWC = list(tokens)
len(words_for_LIWC)


# In[116]:


#Create the LIWC file
df = dict(enumerate(words_for_LIWC))
pd.Series(df).to_csv('LIWC.csv')


# In[117]:


embeddings = {}

for word in tokens:
    if(word in model.wv):
        embeddings[word] = model.wv[word]
    else :
        embeddings[word] = np.random.normal(0, 0.1, 300)


# In[119]:


vocabulary = len(tokens)
vocab_size = vocabulary
word2id = dict(zip(tokens,range(1,vocab_size+1)))
reversed_dictionary = dict(zip(word2id.values(), word2id.keys()))


# In[125]:


#construct embedding matrix
embedding_matrix  = np.zeros((vocab_size+1,300))
for i in range(1,vocabulary):
    embedding_matrix[i] = embeddings[reversed_dictionary[i]]


# In[126]:


embedding_matrix.shape


# In[130]:


complete_string = ' '.join(words)
split_string = complete_string.split('eos')
#remove last string which is a space 
split_string = split_string[:-1]

#reconstruct
split_data = np.array(split_string)
clean = [sentence[1:-1] for sentence in split_data[1:]]

clean.insert(0, split_data[0][:-1])
split_data = np.array(clean)


# In[272]:


utterances = []
l = []
for w in words:
    if(w == 'eos'):
        utterances.append(l)
        l = []
    else:
        l.append(w)


# In[273]:


train, test_tmp = sklearn.model_selection.train_test_split(utterances,test_size = 0.25)
test, valid = sklearn.model_selection.train_test_split(test_tmp, test_size = 0.4)


# In[274]:


for sentence in train:
    sentence.append('eos')
    
for sentence in valid:
    sentence.append('eos')


# In[275]:


def convert_data_to_id(train_data):
    return [word2id[w] for w in train_data] 


# In[276]:


train_data = list(itertools.chain.from_iterable(train))
valid_data = list(itertools.chain.from_iterable(valid))

train_data = convert_data_to_id(train_data)
valid_data = convert_data_to_id(valid_data)


# In[278]:


print(train_data[:10])
print(valid_data[:10])


# In[289]:


data_path = './'
num_steps = 20
batch_size = 20
skip_step = num_steps + 1
hidden_size = 200
num_epochs = 19
input_shape = num_steps


train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step)

valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step)



model = Sequential()
model.add(Embedding(vocabulary+1, 300, input_length=num_steps, weights=[embedding_matrix]))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(TimeDistributed(Dense(vocabulary+1)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', perplexity])
model.summary()


# In[291]:


checkpointer = ModelCheckpoint(filepath=data_path + '/Baselinemodel-{epoch:02d}.hdf5', verbose=1)
model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

