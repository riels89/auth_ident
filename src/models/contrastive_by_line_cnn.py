import sys
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
from src.preprocessing import load_data
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda, Conv2D
from tensorflow.keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]


class contrastive_by_line_cnn():

    def __init__(self):

        self.name = "contrastive_by_line_cnn"
        self.dataset_type = "by_line"

    def create_cnn(self, params, index):
        input = keras.Input(batch_shape=(params[index]["batch_size"],
                                         params[index]["max_lines"],
                                         params[index]["max_line_length"],
                                         params[index]['dataset'].len_encoding),
                            name='place_holder_input')

        conv = Conv2D(128, [5, params[index]["max_line_length"]], strides=1, padding="same", activation="relu", name='conv_1')(input)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 10, strides=1, padding="same", activation="relu", name='conv_2')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 4, strides=2, padding="same", activation="relu", name='conv_3')(conv)
        if params[index]['BN']:
            conv = BatchNormalization()(conv)
        conv = Conv1D(128, 2, strides=1, padding="same", activation="relu", name='conv_4')(conv)

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

        cnn = self.create_cnn(params, index)

        cnn1 = cnn(input1)
        cnn2 = cnn(input2)

        output_embedding1 = Dense(params[index]['embedding_size'], name="output_embedding1")(cnn1)
        output_embedding2 = Dense(params[index]['embedding_size'], name="output_embedding2")(cnn2)

        distance = Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])


        model = keras.Model(inputs=(input1, input2), outputs=distance, name=self.name + "-" + str(index))
        model.summary()

        return model
