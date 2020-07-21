
import os
import sys
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import logging
import sklearn
from itertools import product
import pandas as pd
import json
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.preprocessing.split_dataset import split_dataset
from tensorflow.keras import backend as K
from contrastive_cnn import contrastive_cnn
from simpleNN import simpleNN
from simple_lstm import simple_lstm
from cnn_lstm import cnn_lstm
from split_cnn import split_cnn
from largeNN import largeNN
from split_NN import split_NN
from split_lstm import split_lstm
from split_bilstm import split_bilstm
from contrastive_bilstm import contrastive_bilstm
from contrastive_bilstm_v2 import contrastive_bilstm_v2
from contrastive_stacked_bilstm import contrastive_stacked_bilstm
from multi_attention_bilstm import multi_attention_bilstm
from contrastive_cnn import contrastive_cnn
from contrastive_by_line_cnn import contrastive_by_line_cnn
from contrastive_1D_to_2D import contrastive_1D_to_2D
from dilated_conv_by_line import dilated_conv_by_line
from src.data_processing_expt.closed_dataset import closed_dataset

class train_outer:

    def __init__(self, model, experiment_num):
        self.k_cross_val = 5

        # TODO THIS IS DANGEROUS
        self.model = eval(model + "()")
        #from eval(model) import eval(model)
        temp = "models/" + model + "/EXP" + str(experiment_num) + "*" + "/combination-0"
        model_path = glob.glob(temp)[0]

        params_path = model_path + "/../param_dict.json"
        comb_dir = model_path + "/checkpoints"

        params = json.load(open(params_path))
        params = generate_param_grid(params)
        map_params(params)

        # Create inner model
        model = model.create_model(params, 0, None)
        model.compile(optimizer=params[0]['optimizer'],
                      loss=params[0]['loss'],
                      metrics=[accuracy])
        model.summary()

        # Load most recent checkpoint
        files = os.listdir(comb_dir)
        paths = [os.path.join(comb_dir, basename) for basename in files]
        newest_model = max(paths, key=os.path.getctime)
        model.load_weights(newest_model)

        #strip cnn
        layer_name = 'output_embedding'
        intermediate_layer_model = keras.Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        #intermediate_output = intermediate_layer_model.predict(data)


        #split test1 -> train3 + test3
        #gen = closed_dataset(crop_length=params[0]["max_code_len"], k_cross_val=params[0]["k_cross_val"], language=params[0]["language"])
        gen = closed_dataset(crop_length=1200)
        self.X1, self.y1, self.X2, self.y2 = gen.get_dataset()

        #TODO: Validate using train2 and val2 to lock params

        self.outer_model = create_random_forest(params, 1, None)
        #train on train3
        #test on test3
        #train3_labels = pd.factorize(train3['author'])[0]
        #test3_labels = pd.factorize(test3['author'])[0]
        #outer_model = outer_model.fit(train3, train3_labels)
        #outer_model.set_params(outer_model_params)
        #accuracy = outer_model.score(test3, test3_labels)
        #print("Closed set problem accuracy: " + accuracy)

    def train(self):
        score = sklearn.cross_val_score(self.outer_model, self.X1, self.y1, cv=self.k_cross_val)
        return score



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def map_params(params):
    index = 0
    if params[index]['optimizer'] == 'adam':
        kwargs = {}
        if 'lr' in params[index]:
            kwargs['lr'] = params[index]['lr']
        if 'clipvalue' in params[index]:
            kwargs['clipvalue'] = params[index]['clipvalue']
        elif 'clipnorm' in params[index]:
            kwargs['clipnorm'] = params[index]['clipnorm']
        if 'decay' in params[index]:
            kwargs['decay'] = params[index]['decay']
        params[index]['optimizer'] = keras.optimizers.Adam(**kwargs)

    if params[index]['loss'] == 'contrastive':
        params[index]['loss'] = contrastive_loss
        if 'margin' in params[index]:
            margin = params[index]['margin']


def generate_param_grid(params):
    return [dict(zip(params.keys(), values)) for values in product(*params.values())]


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_random_forest(self, params, index, logger):
    input = keras.Input(batch_shape=(params[index]["batch_size"], params[index]["max_code_length"] + 2,
                                     params[index]['dataset'].len_encoding), name='input')
    return sklearn.ensemble.RandomForestClassifier(**params[index])

if __name__ == "__main__":
    trainer = train_outer("contrastive_cnn", "")
    #X, y = pg.gen()