import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import logging
from itertools import product
import json
from simpleNN import simpleNN
from simple_lstm import simple_lstm
from cnn_lstm import cnn_lstm
from datetime import datetime
import pandas as pd
import numpy as np
from src.preprocessing import load_data
from src.preprocessing.combined_dataset import combined_dataset
from src.preprocessing.split_dataset import split_dataset
from src.preprocessing.by_line_dataset import by_line_dataset
from split_cnn import split_cnn
from largeNN import largeNN
from split_NN import split_NN
from tensorflow.keras.callbacks import LambdaCallback
from split_lstm import split_lstm
from split_bilstm import split_bilstm
from contrastive_bilstm import contrastive_bilstm
from contrastive_bilstm_v2 import contrastive_bilstm_v2
from contrastive_stacked_bilstm import contrastive_stacked_bilstm
from multi_attention_bilstm import multi_attention_bilstm
from contrastive_cnn import contrastive_cnn
from tensorflow.keras import backend as K

from src import TRAIN_LEN, VAL_LEN, SL


class trainer:

    def __init__(self, model, expirement_name, expirement_num,
                 date=datetime.now().strftime("%m-%d-%y")):

        self.logdir = "models" + SL + model.name + SL + "EXP" + str(expirement_num) + '-' \
                      + expirement_name + '-' + date

        assert os.path.isdir(self.logdir), "Dir " + self.logdir + " doesn't exist"

        self.params = json.load(open(self.logdir + "/param_dict.json"))

        self.root_logger = self.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]

        self.root_logger.info("Starting expirement " + expirement_name + '-' + str(expirement_num) \
                              + 'with model ' + model.name)
        self.root_logger.info("Parameter dict: " + str(self.params))
        self.root_logger.info("")

        self.model = model
        self.params = self.generate_param_grid(self.params)

        self.margin = 1

    def train(self):

        parameters = pd.DataFrame(self.params)
        parameters["val_accuracy"] = np.nan
        parameters["val_loss"] = np.nan

        for index in range(len(self.params)):
            curr_log_dir = self.logdir + SL + "combination-" + str(index)
            os.makedirs(curr_log_dir, exist_ok=True)
            os.makedirs(curr_log_dir + '/checkpoints', exist_ok=True)
            logger = self.create_logger(curr_log_dir, index=index)
            logger.addHandler(self.root_handler)

            logger.info('Training with parameter combination ' + str(index))
            logger.info("With parameters: " + str(self.params[index]))
            logger.info("")

            history = self.train_one(index, logger)

            parameters.loc[index, 'val_loss'] = history['val_loss'][0]
            parameters.loc[index, 'val_accuracy'] = history['val_accuracy'][0]
            parameters.iloc[index].to_json(curr_log_dir + '/params.json')

            logger.info("Val loss: " + str(history['val_loss'][0]))
            logger.info("Val accuracy: " + str(history['val_accuracy'][0]))

        parameters.to_csv(self.logdir + "/hyperparameter_matrix.csv")

    def train_one(self, index, logger):

        curr_log_dir = self.logdir + SL + "combination-" + str(index)
        logger.info("Current log dir: " + curr_log_dir)

        training_dataset, val_dataset = self.map_dataset(self.model.dataset_type, index)

        self.map_params(index)

        model = self.model.create_model(self.params, index, logger)

        tensorboard_callback = TensorBoard(log_dir=curr_log_dir,
                                           update_freq=100, profile_batch=0)

        save_model_callback = ModelCheckpoint(curr_log_dir + "/checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                              monitor='val_loss', save_best_only=True, mode='min')

        # def batchOutput(batch, logs):
        #     logger.info("Finished batch: " + str(batch))
        #     logger.info(logs)

        # log_stats_callback = LambdaCallback(on_batch_end=batchOutput)
        # def contrastive_loss(y_true, y_pred):
        #    return - tf.math.log(1 - (y_true * (1-y_pred) + (1 - y_true) * (1+y_pred)) / 2)
        # , "tf_op_layer_Sum": contrastive_loss
        def accuracy(y_true, y_pred):
            '''Compute classification accuracy with a fixed threshold on distances.
            '''
            return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

        model.compile(optimizer=self.params[index]['optimizer'],
                      loss=self.params[index]['loss'],
                      metrics=[accuracy])

        model.summary()

        logger.info('Fit model on training data')

        history = model.fit(training_dataset,
                            validation_data=val_dataset,
                            epochs=self.params[index]['epochs'],
                            steps_per_epoch=TRAIN_LEN // self.params[index]['batch_size'],
                            validation_steps=VAL_LEN // self.params[index]['batch_size'],
                            callbacks=[tensorboard_callback, save_model_callback])
        return history.history

    def generate_param_grid(self, params):
        return [dict(zip(params.keys(), values)) for values in product(*params.values())]

    def create_logger(self, log_dir, name=None, index=None):
        if name is None:
            assert index is not None, "Must specify index in create logger"
            name = 'combination-' + str(index)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_dir + '/' + name + ".log")
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def map_dataset(self, dataset_type, index):

        if dataset_type == "combined":
            dataset = combined_dataset(max_code_length=self.params[index]["max_code_length"],
                                       batch_size=self.params[index]['batch_size'],
                                       binary_encoding=self.params[index]['binary_encoding'])
        elif dataset_type == "split":
            dataset = split_dataset(max_code_length=self.params[index]["max_code_length"],
                                    batch_size=self.params[index]['batch_size'],
                                    binary_encoding=self.params[index]['binary_encoding'])
        elif dataset_type == 'by_line':
            if self.params[index]['loss'] == 'contrastive':
                dataset = by_line_dataset(max_lines=self.params[index]["max_lines"],
                                      max_line_length=self.params[index]["max_line_length"],
                                      batch_size=self.params[index]['batch_size'],
                                      binary_encoding=self.params[index]['binary_encoding'], flip_labels=True)
            else:
                dataset = by_line_dataset(max_lines=self.params[index]["max_lines"],
                                          max_line_length=self.params[index]["max_line_length"],
                                          batch_size=self.params[index]['batch_size'],
                                          binary_encoding=self.params[index]['binary_encoding'])
        self.params[index]['dataset'] = dataset
        return dataset.get_dataset()

    def map_params(self, index):
        if self.params[index]['optimizer'] == 'adam':
            kwargs = {}
            if 'lr' in self.params[index]:
                kwargs['lr'] = self.params[index]['lr']
            if 'clipvalue' in self.params[index]:
                kwargs['clipvalue'] = self.params[index]['clipvalue']
            elif 'clipnorm' in self.params[index]:
                kwargs['clipnorm'] = self.params[index]['clipnorm']
            if 'decay' in self.params[index]:
                kwargs['decay'] = self.params[index]['decay']
            self.params[index]['optimizer'] = keras.optimizers.Adam(**kwargs)

        if self.params[index]['loss'] == 'contrastive':
            self.params[index]['loss'] = self.contrastive_loss
            if 'margin' in self.params[index]:
                self.margin = self.params[index]['margin']


    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)




# trainer(simple_lstm(), "first_runs", 1, date="13-10-19").train()
# trainer(simpleNN(), "dropout_onehot", 5, date="12-10-19").train()
# trainer(cnn_lstm(), "first_runs", 1, "11-25-19").train()
# trainer(split_cnn(), "smaller_cnn", 4, "1-11-20").train()
# trainer(largeNN(), "first_runs", 1, "12-10-19").train()
# trainer(split_NN(), "test_optimizations", 4, "1-16-20").train()
# trainer(split_lstm(), "300_input_size", 3, "1-30-20").train()
# trainer(contrastive_bilstm(), "fixing_error", 2, "2-18-20").train()
# trainer(contrastive_bilstm_v2(), "extra_dense", 4, "4-19-20").train()
# trainer(multi_attention_bilstm(), "smaller_lr", 4, "5-6-20").train()
trainer(contrastive_cnn(), "first_runs", 1, "5-10-20").train()