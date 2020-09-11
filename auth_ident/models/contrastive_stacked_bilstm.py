import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply, Lambda
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
from tensorflow.keras import layers


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]

class ContrastiveStackedBiLSTM():

    def __init__(self):

        self.name = "contrastive_stacked_bilstm"
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

        bi_lstm = Bidirectional(LSTM(512, name='bi_lstm', return_sequences=True))
        lstm = LSTM(512, name='lstm')

        bi_lstm1 = bi_lstm(dense1)
        bi_lstm2 = bi_lstm(dense2)

        lstm1 = lstm(bi_lstm1)
        lstm2 = lstm(bi_lstm2)

        output_embedding1 = Dense(512, name="output_embedding1")(lstm1)
        output_embedding2 = Dense(512, name="output_embedding2")(lstm2)

        # testT = Lambda(test)([output_embedding1, output_embedding2])

        # concat = layers.concatenate([output_embedding1, output_embedding2])
        # distance = Dense(1, activation='sigmoid', name='predictions')(concat)

        # dense1 = Dense(256, activation='relu')(concat)
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])

        # distance = euclidean_distance([output_embedding1, output_embedding2])

        return keras.Model(inputs=[input1, input2], outputs=distance, name=self.name + "-" + str(index))




