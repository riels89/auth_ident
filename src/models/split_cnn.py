import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D, Flatten

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class split_cnn():

    def __init__(self):

        self.name = "split_cnn"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        conv1 = Conv1D(256, 128, strides=1, padding="same", activation="relu", name='1_conv_1')(input1)
        conv1 = Conv1D(256, 128, strides=5, padding="same", activation="relu", name='1_conv_2')(conv1)
        conv1 = Conv1D(256, 64, strides=1, padding="same", activation="relu", name='1_conv_3')(conv1)
        conv1 = Conv1D(256, 32, strides=5, padding="same", activation="relu", name='1_conv_4')(conv1)
        conv1 = Conv1D(256, 8, strides=1, padding="same", activation="relu", name='1_conv_5')(conv1)
        conv1 = Conv1D(512, 1, strides=1, padding="same", activation="relu", name='1_conv_6')(conv1)
        conv1 = Flatten()(conv1)

        conv2 = Conv1D(256, 128, strides=1, padding="same", activation="relu", name='2_conv_1')(input2)
        conv2 = Conv1D(256, 128, strides=5, padding="same", activation="relu", name='2_conv_2')(conv2)
        conv2 = Conv1D(256, 64, strides=1, padding="same", activation="relu", name='2_conv_3')(conv2)
        conv2 = Conv1D(256, 32, strides=5, padding="same", activation="relu", name='2_conv_4')(conv2)
        conv2 = Conv1D(256, 8, strides=1, padding="same", activation="relu", name='2_conv_5')(conv2)
        conv2 = Conv1D(512, 1, strides=1, padding="same", activation="relu", name='2_conv_6')(conv2)
        conv2 = Flatten()(conv2)

        concat = layers.concatenate([conv1, conv2])
        dense1 = Dense(1024, activation='relu', name="dense_1")(concat)
        dense2 = Dense(1024, activation='relu', name="dense_2")(dense1)
        dense3 = Dense(512, activation='relu', name="dense_3")(dense2)

        outputs = Dense(1, activation='sigmoid', name='predictions')(dense3)

        model = keras.Model(inputs=(input1, input2), outputs=outputs, name=self.name + "-" + str(index))
        model.summary()

        return model
