from universal_keras_transformer.transformer import TransformerBlock
from universal_keras_transformer.position import TransformerCoordinateEmbedding, AddCoordinateEncoding
from tensorflow import keras
from tensorflow.keras.layers import Dense, TimeDistributed, Concatenate, Embedding, Average


class UniversalTransformer():
    def __init__(self):

        self.name = "contrastive_cnn"
        self.dataset_type = "split"

    def create_transformer(self, params):

        input = keras.Input(batch_shape=(params["batch_size"],
                                         params["max_code_length"] + 2,
                                         params['input_embedding_size']),
                            name='place_holder_input')

        transformer = TransformerBlock(
            "siamese_transformer",
            params['num_heads'],
            residual_dropout=params['residual_dropout'],
            attention_dropout=params['attention_dropout'],
            activation='gelu',
            use_masking=True)

        add_coordinate_embedding = TransformerCoordinateEmbedding(
            params['transformer_depth'],
            name='coordinate_embedding')

        output = input
        for step in range(params['transformer_depth']):
            coordinate_embedding = add_coordinate_embedding(output, step=step)
            print(coordinate_embedding)
            output = transformer(coordinate_embedding)

        return keras.Model(input, output, name="siamese_transformer")

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

        embedding = Embedding(params['dataset'].len_encoding,
                              output_dim=params['input_embedding_size'],
                              mask_zero=True)
        #embedding = TimeDistributed(embedding)

        embedding1 = embedding(argmax1)
        embedding2 = embedding(argmax2)

        transformer = self.create_transformer(params)
        transformer.summary()

        transformer1 = transformer(embedding1)
        transformer2 = transformer(embedding2)

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
