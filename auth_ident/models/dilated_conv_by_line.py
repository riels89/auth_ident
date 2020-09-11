import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda, Conv2D, Reshape
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]


class DilatedCNNByLine():

    def __init__(self):

        self.name = "dialated_conv_by_line"
        self.dataset_type = "by_line"
        self.input_embedding_size = 32

    def create_cnn(self, params, index):
        input = keras.Input(batch_shape=(params[index]["batch_size"],
                                         params[index]["max_lines"],
                                         params[index]["max_line_length"],
                                         self.input_embedding_size),
                            name='place_holder_input')

        conv = Conv2D(128, [5, params[index]["max_line_length"]], strides=[1, params[index]["max_line_length"]], padding="same", activation="relu", name='conv_1')(input)
        conv = K.squeeze(conv, 2)

        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 10, strides=1, padding="same", activation="relu", name='conv_2')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 4, strides=2, padding="same", activation="relu", dialated=[2], name='conv_3')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 2, strides=1, padding="same", activation="relu", dialated=[4], name='conv_4')(conv)
        conv = Conv1D(128, 2, strides=1, padding="same", activation="relu", dialated=[8], name='conv_5')(conv)

        conv = Flatten()(conv)

        return keras.Model(input, conv)

        
    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params[index]["batch_size"],
                                          params[index]["max_lines"],
                                          params[index]["max_line_length"],
                                          params[index]['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params[index]["batch_size"],
                                          params[index]["max_lines"],
                                          params[index]["max_line_length"],
                                          params[index]['dataset'].len_encoding),
                             name='input_2')

        embedding = Dense(self.input_embedding_size, name='input_embedding')
        embedding = TimeDistributed(embedding)

        embedding1 = embedding(input1)
        embedding2 = embedding(input2)

        cnn = self.create_cnn(params, index)
        cnn.summary()

        cnn1 = cnn(embedding1)
        cnn2 = cnn(embedding2)

        output_embedding = Dense(256, name="output_embedding")

        output_embedding1 = output_embedding(cnn1)
        output_embedding2 = output_embedding(cnn2)

        distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])

        model = keras.Model(inputs=(input1, input2), outputs=distance, name=self.name + "-" + str(index))
        model.summary()

        return model
