import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class cnn_lstm():

    def __init__(self):

        self.name = "cnn_lstm"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        conv1 = Conv1D(255, 47, strides=47, name='embedding')(input1)
        conv2 = Conv1D(255, 47, strides=47, name='embedding')(input2)

        lstm1 = LSTM(255, name='lstm1')(input1, initial_state=conv1)
        lstm2 = LSTM(255, name='lstm2')(input2, initial_state=conv2)

        concat = layers.concatenate([lstm1, lstm2])
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        model = keras.Model(inputs=(input1, input2), outputs=outputs, name=self.name + "-" + str(index))

        return model
