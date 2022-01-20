# CREDITS: https://www.analyticsvidhya.com/blog/2021/12/multiclass-classification-using-transformers/


import pandas as pd

# importing the dataset 
df_train = pd.read_csv('0.2_fine_tuning_pre_trained_model/emotion_dataset/train.txt', header =None, sep =';', names = ['Input','Sentiment'], encoding='utf-8')
df_test = pd.read_csv('0.2_fine_tuning_pre_trained_model/emotion_dataset/test.txt', header = None, sep =';', names = ['Input','Sentiment'],encoding='utf-8')

df_train.head()
""" 
                                               Input Sentiment
0                            i didnt feel humiliated   sadness
1  i can go from feeling so hopeless to so damned...   sadness
2   im grabbing a minute to post i feel greedy wrong     anger
3  i am ever feeling nostalgic about the fireplac...      love
4                               i am feeling grouchy     anger """

# Converting our Sentiment column into Categorical data
encoded_dict = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}

df_train['Sentiment'] = df_train.Sentiment.map(encoded_dict)
df_test['Sentiment'] = df_test.Sentiment.map(encoded_dict)

df_train.head()
""" 
                                               Input  Sentiment
0                            i didnt feel humiliated          4
1  i can go from feeling so hopeless to so damned...          4
2   im grabbing a minute to post i feel greedy wrong          0
3  i am ever feeling nostalgic about the fireplac...          3
4                               i am feeling grouchy          0 """

from tensorflow.keras.utils import to_categorical
# converting our integer coded Sentiment column into categorical data(matrix)

y_train = to_categorical(df_train.Sentiment)
y_test = to_categorical(df_test.Sentiment)

y_train
""" 
array([[0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0., 0.],
       ...,
       [0., 0., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0.]], dtype=float32) """

# We have successfully processed our Sentiment column( target); now, it’s time to process our input text data 
# using a tokenizer.

import transformers

# Loading Model and Tokenizer from the transformers package 

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

# Tokenize the input (takes some time) 
# here tokenizer using from bert-base-cased
x_train = tokenizer(text=df_train.Input.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

x_test = tokenizer(text=df_test.Input.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']

""" Model Building
Importing necessary libraries.
 """
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
# We are using functional API to design our model.

max_len = 70
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(6,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# ref: https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
# ref: https://www.tensorflow.org/install/source#gpu

train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
    ),
  epochs=2,
    batch_size=36
)

predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
predicted_raw[0]

import numpy as np
y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = df_test.Sentiment

from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))

texts = input(str('input the text'))
x_val = tokenizer(text=texts,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 

validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100

for key , value in zip(encoded_dict.keys(),validation[0]):
    print(key,value)