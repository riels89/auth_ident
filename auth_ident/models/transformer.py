from tensorflow import keras
from tensorflow.keras.layers import Dense, TimeDistributed, Concatenate, Embedding, Average
from keras_transformer import get_model


class Transformer():
    def __init__(self):

        self.name = "contrastive_cnn"
        self.dataset_type = "split"

    def create_transformer(self, params):

        transformer = get_model(
            token_num=params['dataset'].len_encoding,
            embed_dim=params['input_embedding_size'],
            encoder_num=params['encoder_num'],
            decoder_num=params['decoder_num'],
            head_num=params["num_heads"],
            hidden_dim=params["d_model"],
            attention_activation=None,
            dropout_rate=params["dropout_rate"],
            embed_weights=None
        )

        return transformer

    def create_model(self, params, index, logger):

        input1 = keras.Input(batch_shape=(params["batch_size"],
                                          params["max_code_length"] + 2,
                                          params['dataset'].len_encoding),
                             name='input_1')
        input2 = keras.Input(batch_shape=(params["batch_size"],
                                          params["max_code_length"] + 2,
                                          params['dataset'].len_encoding),
                             name='input_2')

        argmax1 = keras.backend.argmax(input1, axis=2)
        argmax2 = keras.backend.argmax(input2, axis=2)

        transformer = self.create_transformer(params)
        transformer.summary()

        transformer1 = transformer([argmax1, argmax1])
        transformer2 = transformer([argmax2, argmax2])

        print(transformer1)
        average1 = keras.backend.mean(transformer1, axis=1)
        average2 = keras.backend.mean(transformer2, axis=1)
        print(average1)
        hidden = Dense(512, name="hidden")
        output_embedding = Dense(params['embedding_size'], name="output_embedding")

        output_embedding1 = output_embedding(hidden(average1))
        output_embedding2 = output_embedding(hidden(average2))

        embs = Concatenate(axis=0)([output_embedding1, output_embedding2])
        print(embs)

        model = keras.Model(inputs=(input1, input2),
                            outputs=embs,
                            name=self.name + "-" + str(index))
        model.summary()

        return model
