import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
import tensorflow as tf


class ContrastiveBiLSTM():

    def __init__(self):

        self.name = "contrastive_bilstm"
        self.dataset_type = "split"

    def cosine_distance(self, lstm1, lstm2):
        normalize_a = backend.l2_normalize(lstm1, 1)        
        normalize_b = backend.l2_normalize(lstm2, 1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)

        # cos = cosine_similarity(lstm1, lstm2, axis=1)
        return cos_similarity

        
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
        # dense1 = Dense(256, activation='relu')(concat)
        outputs = Dense(1, activation='sigmoid', name='predictions')(concat)

        cos_similarity = self.cosine_distance(lstm1, lstm2)

        return keras.Model(inputs=[input1, input2], outputs=[outputs], name=self.name + "-" + str(index))




