import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))

import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
import talos as ta

class simpleNN():

    def __init__(self):
        self.name = "simpleNN"


    def create_model(self, params, index, logger):

        inputs = keras.Input(shape=(load_data.max_code_length * 2 + 1,
                                    load_data.binary_encoding_len),
                             name='code')
        x = layers.Flatten()(inputs)
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = keras.layers.Dropout(params[index]['dropout'])(x)
        x = layers.Dense(512, activation='relu', name='dense_2')(x)
        x = keras.layers.Dropout(params[index]['dropout'])(x)
        x = layers.Dense(512, activation='relu', name='dense_3')(x)
        x = keras.layers.Dropout(params[index]['dropout'])(x)
        x = layers.Dense(512, activation='relu', name='dense_4')(x)
        x = keras.layers.Dropout(params[index]['dropout'])(x)
        x = layers.Dense(512, activation='relu', name='dense_5')(x)
        x = keras.layers.Dropout(params[index]['dropout'])(x)
        x = layers.Dense(256, activation='relu', name='dense_6')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)

        return keras.Model(inputs=inputs, outputs=outputs, name=self.name + "-" + str(index))
