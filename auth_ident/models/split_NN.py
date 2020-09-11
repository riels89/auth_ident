import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D, Flatten


class SplitNN():

    def __init__(self):

        self.name = "split_NN"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_1')
        input1_flat = layers.Flatten()(input1)

        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"],
                             params[index]['dataset'].len_encoding),
                             name='input_2')
        input2_flat = layers.Flatten()(input2)

        dense1 = layers.Dense(1024, activation='relu', name='dense1_1')(input1_flat)
        dense1 = keras.layers.Dropout(params[index]['dropout'])(dense1)
        dense1 = layers.Dense(1024, activation='relu', name='dense1_2')(dense1)
        dense1 = keras.layers.Dropout(params[index]['dropout'])(dense1)
        dense1 = layers.Dense(1024, activation='relu', name='dense1_3')(dense1)

        dense2 = layers.Dense(1024, activation='relu', name='dense2_1')(input2_flat)
        dense2 = keras.layers.Dropout(params[index]['dropout'])(dense2)
        dense2 = layers.Dense(1024, activation='relu', name='dense2_2')(dense2)
        dense2 = keras.layers.Dropout(params[index]['dropout'])(dense2)
        dense2 = layers.Dense(1024, activation='relu', name='dense2_3')(dense2)


        concat = layers.concatenate([dense1, dense2])
        dense1 = Dense(2024, activation='relu', name="dense3_1")(concat)
        dense2 = Dense(1024, activation='relu', name="dense3_2")(dense1)
        dense3 = Dense(512, activation='relu', name="dense3_3")(dense2)
        dense4 = Dense(256, activation='relu', name="dense3_4")(dense3)

        outputs = Dense(1, activation='sigmoid', name='predictions')(dense4)

        model = keras.Model(inputs=(input1, input2), outputs=outputs, name=self.name + "-" + str(index))
        model.summary()

        return model
