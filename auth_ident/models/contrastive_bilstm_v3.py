import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional, multiply
from tensorflow.keras.losses import cosine_similarity
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Dropout, Concatenate
import tensorflow as tf


class ContrastiveBilstmV3():

    def __init__(self):

        self.name = "contrastive_bilstm_v3"
        self.dataset_type = "split"


    def create_model(self, params, index, logger):

        input_left = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_1')
        input_right = keras.Input(
            batch_shape=(None,
                         params["max_code_length"]),
            name='input_2')

        embedding = Embedding(params['dataset'].len_encoding,
                              params['input_embedding_size'],
                              input_length=params["max_code_length"])

        embedding_left = embedding(input_left)
        embedding_right = embedding(input_right)

        lstm1 = Bidirectional(LSTM(128, name='lstm1', return_sequences=True, dropout=params['dropout']), merge_mode="concat")
        lstm2 = Bidirectional(LSTM(128, name='lstm2'), merge_mode="concat", name='output_embedding')
        
        lstm1_left = lstm1(embedding_left)
        lstm1_right = lstm1(embedding_right)

        lstm2_left = lstm2(lstm1_left)
        lstm2_right = lstm2(lstm1_right)
        
        lstm_concat = Concatenate(axis=0)([lstm2_left, lstm2_right])

        return keras.Model(inputs=[input_left, input_right], outputs=lstm_concat, name=self.name + "-" + str(index))

