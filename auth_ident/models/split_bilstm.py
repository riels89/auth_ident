import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional


class SplitBilstm():

    def __init__(self):

        self.name = "split_bilstm"
        self.dataset_type = "split"
        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_1')

        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding),
                             name='input_2')

        embedding = Dense(32, name='embedding1')

        dense1 = embedding(input1)
        dense2 = embedding(input2)

        lstm = Bidirectional(LSTM(512, name='lstm1'))

        lstm1 = lstm(dense1)
        lstm2 = lstm(dense2)

        concat = layers.concatenate([lstm1, lstm2])
        dense1 = Dense(256, activation='relu')(concat)
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        return keras.Model(inputs=[input1, input2], outputs=outputs, name=self.name + "-" + str(index))




