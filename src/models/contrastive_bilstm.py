import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.losses import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class contrastive_bilstm():

    def __init__(self):

        self.name = "contrastive_bilstm"
        self.dataset_type = "split"

    def cosine_distance(self, lstm1, lstm2):
        cos = cosine_similarity(lstm1, lstm2, axis=1)
        return cos

        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_1')

        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        embedding = Dense(32, name='embedding1')

        dense1 = embedding(input1)
        dense2 = embedding(input2)

        lstm = Bidirectional(LSTM(512, name='lstm1'))

        lstm1 = lstm(dense1)
        lstm2 = lstm(dense2)

        concat = layers.concatenate([lstm1, lstm2])
        dense1 = Dense(256, activation='relu')(concat)
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        cos_similarity = self.cosine_distance(lstm1, lstm2)

        return keras.Model(inputs=[input1, input2], outputs=[outputs, cos_similarity], name=self.name + "-" + str(index))




