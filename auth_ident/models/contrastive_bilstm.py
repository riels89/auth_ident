import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Dropout, Concatenate
import tensorflow as tf


class ContrastiveBiLSTM():

    def __init__(self):

        self.name = "contrastive_bilstm"
        self.dataset_type = "split"


    def create_model(self, params, index, logger):

        input1 = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_1')
        input2 = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_2')

        embedding = Embedding(params['dataset'].len_encoding,
                              params['input_embedding_size'],
                              input_length=params["max_code_length"])

        embedding1 = embedding(input1)
        embedding2 = embedding(input2)

        lstm = Bidirectional(LSTM(256, name='lstm1'), merge_mode="concat", name="output_embedding")

        lstm1 = lstm(embedding1)
        lstm2 = lstm(embedding2)
        lstm_concat = Concatenate(axis=0)([lstm1, lstm2])

        #output_embedding = Dense(params['embedding_size'], 
        #                        name="output_embedding")
        #emb1 = output_embedding(lstm1)
        #emb2 = output_embedding(lstm2)

        #emb_concat = Concatenate(axis=0)([emb1, emb2])

        return keras.Model(inputs=[input1, input2], outputs=lstm_concat, name=self.name + "-" + str(index))




