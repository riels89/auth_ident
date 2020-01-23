import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class split_lstm():

    def __init__(self):

        self.name = "split_lstm"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_1')

        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        dense1 = Dense(256, name='embedding1')(input1)
        dense2 = Dense(256, name='embedding2')(input2)

        lstm1 = LSTM(512, name='lstm1')(dense1)
        lstm2 = LSTM(512, name='lstm2')(dense2)

        concat = layers.concatenate([lstm1, lstm2])
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        return keras.Model(inputs=[input1, input2], outputs=outputs, name=self.name + "-" + str(index))




