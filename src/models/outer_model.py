
import os
import sys
import tensorflow.keras as keras
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from itertools import product
import json
import glob
from datetime import datetime
import numpy as np

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
from simclr.objective import add_contrastive_loss
from src.data_processing_expt.simclr_dataset import SimCLRDataset

class outer_model:

    def __init__(self, exp_name, exp_num, date=datetime.now().strftime("%m-%d-%y")):
        self.logdir = "models/outer_model/" + "EXP" + str(exp_num) + '-' + exp_name + '-' + date
        assert os.path.isdir(self.logdir), "Dir " + self.logdir + " doesn't exist"
        self.params = json.load(open(self.logdir + "/param_dict.json"))
        self.params = generate_param_grid(self.params)

        # also populates some params
        encoder=self.load_encoder(self.params[0]["encoder_model"], self.params[0]["encoder_exp"])

        #strip cnn
        layer_name = 'output_embedding'
        intermediate_layer_model = keras.Model(inputs=encoder.input,
                                         outputs=encoder.get_layer(layer_name).output)
        intermediate_layer_model.summary()
        #exit()
        #encoder_embedding = intermediate_layer_model.predict(data)

        #self.params[0]["max_code_length"]=10
        #split test1 -> train3 + test3
        #gen = closed_dataset(crop_length=params[0]["max_code_len"], k_cross_val=params[0]["k_cross_val"], language=params[0]["language"])
        gen = closed_dataset(crop_length=self.params[0]["max_code_length"], k_cross_val=self.params[0]["k_cross_val"],
                             language=self.params[0]["language"])
        self.X1, self.y1, self.X2, self.y2 = gen.get_datasets()

        self.X1 = intermediate_layer_model.predict(self.X1, batch_size=self.params[0]["batch_size"])
        self.X2 = intermediate_layer_model.predict(self.X2, batch_size=self.params[0]["batch_size"])

        #TODO: Validate using train2 and val2 to lock params

        self.outer_model = create_random_forest(self.params, 0, None)
        #train on train3
        #test on test3
        #train3_labels = pd.factorize(train3['author'])[0]
        #test3_labels = pd.factorize(test3['author'])[0]
        #outer_model = outer_model.fit(train3, train3_labels)
        #outer_model.set_params(outer_model_params)
        #accuracy = outer_model.score(test3, test3_labels)
        #print("Closed set problem accuracy: " + accuracy)

    def train_and_val(self):
        print("train_and_val", flush=True)
        score = cross_val_score(self.outer_model, self.X1, self.y1, verbose=0, cv=self.params[0]["k_cross_val"])
        return score

    def train_and_test(self):
        print("train_and_test", flush=True)
        test_proportion = 1/self.params[0]["k_cross_val"]
        if test_proportion < .1:
            test_proportion = .1
        X_train, X_test, y_train, y_test = train_test_split(self.X2, self.y2, test_size=test_proportion,
                                                            stratify=self.y2)
        #TODO Sanity check
        for i in y_test:
            assert i in y_train
        num_auth=len(np.unique(np.array(y_test)))
        print("Num_file in test_set: ", len(y_test))
        print("Num_auth in test set: ", num_auth)

        self.outer_model.fit(X_train, y_train)
        return self.outer_model.score(X_test, y_test)


    def load_encoder(self, encoder_name, encoder_exp, comb_num=0):
        # TODO THIS IS DANGEROUS
        model = eval(encoder_name + "()")
        # from eval(model) import eval(model)
        temp = "models/" + encoder_name + "/EXP" + str(encoder_exp) + "*" + "/combination-" + str(comb_num)
        model_path = glob.glob(temp)[0]

        params_path = model_path + "/../param_dict.json"
        # print(str(params_path))
        comb_dir = model_path + "/checkpoints"

        params = json.load(open(params_path))
        params = generate_param_grid(params)
        self.map_dataset(model.dataset_type, comb_num, params)
        map_params(params)
        # print(str(params))

        # Create inner model
        encoder = model.create_model(params, comb_num, None)

        encoder.compile(optimizer=params[0]['optimizer'],
                           loss=params[0]['loss'],
                           metrics=[accuracy])
        #encoder.summary()
        #print("n\n\END\n\n")

        # Load most recent checkpoint
        files = os.listdir(comb_dir)
        paths = [os.path.join(comb_dir, basename) for basename in files]
        newest_model = max(paths, key=os.path.getctime)
        encoder.load_weights(newest_model)

        self.params[0]["language"] = params[comb_num]["language"]
        self.params[0]["binary_encoding"] = params[comb_num]["binary_encoding"]
        self.params[0]["max_code_length"] = params[comb_num]["max_code_length"]
        self.params[0]["embedding_size"] = params[comb_num]["embedding_size"]
        self.params[0]["batch_size"] = params[comb_num]["batch_size"]

        return encoder

    def map_dataset(self, dataset_type, index, params):

        if dataset_type == "split":
            dataset = split_dataset(max_code_length=params[index]["max_code_length"],
                                    batch_size=params[index]['batch_size'],
                                    binary_encoding=params[index]['binary_encoding'],
                                    language=params[index].get('language'))
        elif dataset_type == "simclr":
            dataset = SimCLRDataset(max_code_length=self.params[index]["max_code_length"],
                                    batch_size=self.params[index]['batch_size'],
                                    binary_encoding=self.params[index]['binary_encoding'],
                                    language=self.params[index].get('language'))
        else :
            print("Error: Only split and simclr datasets are supported outer_model.map_dataset")
            exit(1)
        params[index]['dataset'] = dataset

    def map_params(self):
        index = 0
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
            self.params[index]['loss'] = contrastive_loss
            if 'margin' in self.params[index]:
                self.margin = self.params[index]['margin']

        if self.params[index]['loss'] == 'simclr':
            self.params[index]['loss'] = simclr_loss
            if 'temperature' in self.params[index]:
                self.temperature = self.params[index]['temperature']

    def simclr_loss(self, y_true, y_pred):
        '''SimCLR loss from Chen-et-al.'20
           http://arxiv.org/abs/2002.05709
        '''
        print("\nshape:\n", y_pred.shape, flush=True)
        return add_contrastive_loss(y_pred, temperature=self.temperature)[0]

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
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

    if params[index]['loss'] == 'simclr':
        params[index]['loss'] = simclr_loss
        if 'temperature' in params[index]:
            temperature = params[index]['temperature']


def generate_param_grid(params):
    return [dict(zip(params.keys(), values)) for values in product(*params.values())]


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_random_forest(params, index, logger):
    return KNeighborsClassifier(n_jobs=-1)
    #return SVC()
    #return RandomForestClassifier(n_jobs=-1, verbose=0, warm_start=True, min_samples_leaf=5)

if __name__ == "__main__":
    model = outer_model("placeholder", 1, "7-22-20")
    score = model.train_and_val()
    print("Train val scores: ", score)
    test_acc = model.train_and_test()
    print("Test accuracy: ", test_acc)