import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class bi_input_lstm():

    def __init__(self):

        self.name = "bi_input_lstm"
        self.input_shape = "split"
        
    def create_model(self, params, index, logger):

        inputs = keras.Input(shape=(params[index]["max_code_length"] * 2 + 1,
                                    params[index]['dataset'].len_encoding),
                             name='code')
        
        dense1 = Dense(256, name='embedding')(inputs[:load_data.max_code_length])
        dense2 = Dense(256, name='embedding')(inputs[load_data.max_code_length + 1:])

        lstm1 = LSTM(512, name='lstm')(dense1)
        lstm2 = LSTM(512, name='lstm')(dense2)

        concat = layers.concat([lstm1, lstm2])
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name + "-" + str(index))




