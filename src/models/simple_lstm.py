import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM


class simple_lstm():

    def __init__(self):

        self.name = "simple_lstm"
        self.dataset_type = "combined"

    def create_model(self, params, index, logger):

        inputs = keras.Input(shape=(params[index]["max_code_length"] * 2 + 1,
                                    load_data.len_encoding),
                             name='input')
        model = Dense(256, name='embedding')(inputs)
        model = LSTM(256, name='lstm')(model)
        outputs = Dense(1, activation='sigmoid', name='predictions')(model)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name + "-" + str(index))
