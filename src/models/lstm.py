import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))

import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

class simple_lstm():

    def __init__(self):

        self.name = "simple_lstm"

    def create_model(self, params, index, logger):

        inputs = keras.Input(shape=(load_data.max_code_length * 2 + 1,
                                    load_data.binary_encoding_len),
                             name='code')
        model = Dense(256, name='embedding')(inputs)
        model = LSTM(64, name='lstm')(model)
        outputs = Dense(1, activation='sigmoid', name='predictions')(model)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name + "-" + str(index))




