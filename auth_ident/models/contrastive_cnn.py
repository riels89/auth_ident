import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Embedding, TimeDistributed, MaxPool1D
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda, Add
from tensorflow.keras import backend as K
import tensorflow as tf


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]


class ContrastiveCNN():
    def __init__(self):

        self.name = "contrastive_cnn"
        self.dataset_type = "split"

    def create_cnn(self, params):

        l2 = tf.keras.regularizers.l2(params['l2'])

        input = keras.Input(batch_shape=(None,
                                         params["max_code_length"],
                                         params["input_embedding_size"]),
                            name='place_holder_input')

        conv = Conv1D(256,
                      7,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_1')(input)
        if params['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_1")(conv)

        conv = Conv1D(256,
                      7,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_2')(conv)
        if params['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_2")(conv)

        conv = Conv1D(256,
                      3,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_3')(conv)
        if params['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256,
                      3,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_4')(conv)
        if params['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256,
                      3,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_5')(conv)
        if params['BN']:
            conv = BatchNormalization()(conv)

        conv = Conv1D(256,
                      3,
                      strides=1,
                      padding="same",
                      activation="relu",
                      kernel_regularizer=l2,
                      name='conv_6')(conv)
        if params['BN']:
            conv = BatchNormalization()(conv)
        conv = MaxPool1D(3, padding="valid", name="max_pool_3")(conv)

        conv = Flatten()(conv)

        return keras.Model(input, conv, name="siamese_cnn")

    def create_model(self, params, index, logger):

        input1 = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_1')
        input2 = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_2')

        #embedding = Dense(self.input_embedding_size, name='input_embedding')
        #embedding = TimeDistributed(embedding)
        embedding = Embedding(params['dataset'].len_encoding,
                              params['input_embedding_size'],
                              input_length=params["max_code_length"])
        embedding1 = embedding(input1)
        embedding2 = embedding(input2)
        
        if params["bias"]:
            print("Adding bias")
            bias = Embedding(params['dataset'].len_encoding,
                             1,
                             input_length=params["max_code_length"],
                             embeddings_initializer="zeros",
                             name="embeddings_bias")
            bias1 = bias(input1)
            bias2 = bias(input2)
            add = Add()

            embedding1 = add([embedding1, bias1])
            embedding2 = add([embedding2, bias2])

        cnn = self.create_cnn(params)
        cnn.summary()

        cnn1 = cnn(embedding1)
        cnn2 = cnn(embedding2)

        l2 = tf.keras.regularizers.l2(params['l2'])

        non_linearity_1 = Dense(1028,
                                activation="relu",
                                kernel_regularizer=l2,
                                name="non_linearity_1")
        non_linearity_2 = Dense(1028,
                                activation="relu",
                                kernel_regularizer=l2,
                                name="non_linearity_2")

        output_embedding = Dense(params['embedding_size'], 
                                kernel_regularizer=l2,
                                name="output_embedding")

        dropout = Dropout(rate=params['dropout'])
        non_linearity1 = dropout(non_linearity_1(cnn1))
        non_linearity2 = dropout(non_linearity_1(cnn2))

        non_linearity1 = dropout(non_linearity_2(non_linearity1))
        non_linearity2 = dropout(non_linearity_2(non_linearity2))

        output_embedding1 = output_embedding(non_linearity1)
        output_embedding2 = output_embedding(non_linearity2)

        embs = Concatenate(axis=0)([output_embedding1, output_embedding2])
        #distance = Lambda(
        #    euclidean_distance,
        #    output_shape=eucl_dist_output_shape,
        #    name='distance')([output_embedding1, output_embedding2])

        model = keras.Model(inputs=(input1, input2),
                            outputs=embs,
                            name=self.name + "-" + str(index))
        model.summary()

        return model
