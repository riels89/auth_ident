
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import logging
from itertools import product
import json
import glob
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

class train_outer:

    def __init__(self, model, experiment_num):
        # TODO THIS IS DANGEROUS
        self.model = eval(model + "()")
        temp = "models/" + model + "/EXP" + experiment_num + "*" + "/combination-0"
        model_path = glob.glob(temp)[0]

        params_path = model_path + "/../param_dict.json"
        comb_dir = model_path + "/checkpoints"

        params = json.load(open(params_path))
        params = generate_param_grid(params)
        map_params(params)
        print(params)
        print(params[0])
        print(params[1])
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

    def train(self):
        print("train")
        #train



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




file1 = sys.argv[3]
file2 = sys.argv[4]
matching = sys.argv[5]

dataset_builder = split_dataset(max_code_length=params[0]["max_code_length"],
                                batch_size=1,
                                binary_encoding=params[0]['binary_encoding'])
params[0]['dataset'] = dataset_builder
dataset = dataset_builder.encode_files(tf.convert_to_tensor([[file1, file2]]), tf.convert_to_tensor([matching]))

# Create model
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

# Evaluate the model
predict = model.predict_on_batch(dataset)

if predict[0,0] <= .5:
    print("Not a match")
else:
    print("Match")

print(predict)