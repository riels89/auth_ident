import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply, Lambda, Flatten
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
from tensorflow.keras import layers
from keras_multi_head import MultiHeadAttention


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return [shape1[0], 1]

class MultiHeadAttentionBiLSTM():

    def __init__(self):

        self.name = "multi_attention_bilstm"
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

        lstm = Bidirectional(LSTM(512, name='lstm', return_sequences=True))

        if params[index]["attention"]: 
            attention = MultiHeadAttention(head_num=2, name="multi_head_attention")

        lstm1 = lstm(dense1)
        lstm2 = lstm(dense2)


        if params[index]["attention"]: 
            lstm1 = Flatten()(attention(lstm1))
            lstm2 = Flatten()(attention(lstm2))

        output_embedding = Dense(512, name="output_embedding")

        output_embedding1 = output_embedding(lstm1)
        output_embedding2 = output_embedding(lstm2)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape, name='distance')([output_embedding1, output_embedding2])

        return keras.Model(inputs=[input1, input2], outputs=distance, name=self.name + "-" + str(index))




