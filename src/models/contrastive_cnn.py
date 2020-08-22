import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, TimeDistributed, MaxPool1D
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]


class contrastive_cnn():

    def __init__(self):

        self.name = "contrastive_cnn"
        self.dataset_type = "simclr"
        self.input_embedding_size = 32

    def create_cnn(self, params, index):
        input = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                            self.input_embedding_size),
                            name='place_holder_input')

        conv = Conv1D(256, 7, strides=1, padding="same", activation="relu", name='conv_1')(input)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_1")(conv)

        conv = Conv1D(256, 7, strides=1, padding="same", activation="relu", name='conv_2')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_2")(conv)

        conv = Conv1D(256, 3, strides=1, padding="same", activation="relu", name='conv_3')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256, 3, strides=1, padding="same", activation="relu", name='conv_4')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256, 3, strides=1, padding="same", activation="relu", name='conv_5')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256, 3, strides=1, padding="same", activation="relu", name='conv_6')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_3")(conv)

        conv = Flatten()(conv)

        return keras.Model(input, conv, name="siamese_cnn")

        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding))
        input2 = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                             params[index]['dataset'].len_encoding))

        embedding = Dense(self.input_embedding_size, name='input_embedding')
        embedding = TimeDistributed(embedding)

        embedding1 = embedding(input1)
        embedding2 = embedding(input2)

        cnn = self.create_cnn(params, index)
        cnn.summary()

        cnn1 = cnn(embedding1)
        cnn2 = cnn(embedding2)

        non_linearity_1 = Dense(1048, activation="relu", name="non_linearity_1")
        non_linearity_2 = Dense(1048, activation="relu", name="non_linearity_2")

        output_embedding = Dense(256, name="output_embedding")

        non_linearity1 = non_linearity_1(cnn1)
        non_linearity2 = non_linearity_1(cnn2)

        non_linearity1 = non_linearity_2(non_linearity1)
        non_linearity2 = non_linearity_2(non_linearity2)

        output_embedding1 = output_embedding(non_linearity1)
        output_embedding2 = output_embedding(non_linearity2)

        distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])


        model = keras.Model(inputs=(input1, input2), outputs=distance, name=self.name + "-" + str(index))
        model.summary()

        return model
