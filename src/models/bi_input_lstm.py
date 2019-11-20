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

        inputs = keras.Input(shape=(2, load_data.max_code_length,
                                    load_data.len_encoding),
                             name='code')
        
        model1 = Dense(256, name='embedding')(inputs[0])
        model2 = Dense(256, name='embedding')(inputs[1])

        model1 = LSTM(512, name='lstm')(model1)
        model2 = LSTM(512, name='lstm')(model2)

        concat = layers.concat([model1, model2])
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name + "-" + str(index))




