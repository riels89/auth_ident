import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply, Lambda, Conv1D, Flatten
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from ArcFace import ArcMarginPenaltyLogists

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class arcface_cnn():

    def __init__(self):

        self.name = "arcface_cnn"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                            params[index]['dataset'].len_encoding),
                            name='input')

        conv = Conv1D(256, 128, strides=1, padding="same", activation="relu", name='conv_1')(input)
        bn = keras.layers.BatchNormalization(name="BN_1")(conv)
        conv = Conv1D(256, 128, strides=5, padding="same", activation="relu", name='conv_2')(bn)
        bn = keras.layers.BatchNormalization(name="BN_2")(conv)
        conv = Conv1D(256, 1, strides=1, padding="same", activation="relu", name='conv_3')(bn)
        bn = keras.layers.BatchNormalization(name="BN_3")(conv)
        conv = Flatten()(conv)


        embedding = Dense(512, name="output_embedding1")(conv)

        logits = ArcMarginPenaltyLogists()(embedding)

        return keras.Model(inputs=input, outputs=logits, name=self.name + "-" + str(index))




