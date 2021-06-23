from sklearn.model_selection import cross_val_score
from auth_ident.datasets import ClosedDataset

from itertools import product
from tensorflow.keras import backend as K
from auth_ident import GenericExecute
from auth_ident import param_mapping
import os
import pandas as pd
from auth_ident.utils import get_embeddings, get_data, get_model
import time
import tensorflow as tf

from auth_ident.models import GenericSecondaryClassifier 
from auth_ident.datasets import ClosedDataset

from sklearn.model_selection import cross_val_score, train_test_split, KFold

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn.metrics

from histogram_utils import indep_roll, log_odds_same_averaged, log_odds_same_independent, calc_scores, create_pdfs


from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

import matplotlib
matplotlib.rcParams.update({'font.size': 8})
matplotlib.rcParams.update({'font.sans-serif':['Arial']})


class TrainHistogramVerifier(GenericExecute):
    """
    Is the training class for the secondary classifiers.

    Uses a separate `secondary` dictionary in json, but still needs the 
    `contrastive` param dict to use the correct contrastive models and save the
    models to the correct folders.

    """
    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
          # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
          try:
            #tf.config.experimental.set_virtual_device_configuration(
            #    gpus[0],
            #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        super().__init__()

    def execute_one(self, contrastive_params, combination, logger):
        """
        Loops over all of the secondary parameter combinations for each
        contrastive combination.
        """

        
        if self.mode == 'test':
            file_param = "test_data"
        else:
            file_param = "val_data"

        output_layer_name = 'output_embedding'



        data_file = contrastive_params[file_param]

        train_data, train_labels = get_embeddings(
            params=contrastive_params,
            dataset=ClosedDataset,
            max_authors=1600,
            k_cross_val=9,
            output_layer_name=output_layer_name,
            data_file=data_file,
            combination=combination,
            logger=logger,
            logdir=self.logdir,
            normalize= output_layer_name == "output_embedding")



        print(train_data.shape)

        self.full_log_dir = os.path.join(self.logdir, "combination-" + str(combination))


        self.run(train_data, train_labels)


    def run(self, X, y):
        num_authors = len(np.unique(y))

        assert(y[0] == y[num_authors] == y[2 * num_authors])

        import pickle

        same_hist_path = os.path.join(self.full_log_dir, "same_dist.pkl")
        diff_hist_path = os.path.join(self.full_log_dir, "diff_dist.pkl")
        
        
        if self.mode == 'train':
            # Hack... Manually set things up to create the distributions using
            # the validation split, then change the flag to use
            # distribution using the test split.
            same_distribution, diff_distribution = create_pdfs(X, y)
           
            pickle.dump(same_distribution, open(same_hist_path, "wb" ))
            pickle.dump(diff_distribution, open(diff_hist_path, "wb" ))
        else:
            same_distribution = pickle.load(open(same_hist_path, "rb"))
            diff_distribution = pickle.load(open(diff_hist_path, "rb"))
            
        # PLOT THE HISTOGRAMS
        x = np.linspace(same_distribution.support()[0],
                        same_distribution.support()[1], 100, endpoint=False)

        plt.figure(figsize=(2.5, 2.0), dpi=220)
      
        plt.plot(x, same_distribution.pdf(x), color='b', lw=1.0)
        plt.fill_between(x, same_distribution.pdf(x), y2=0, color='b', alpha=.2)

        x = np.linspace(diff_distribution.support()[0],
                        diff_distribution.support()[1], 100, endpoint=False)
        
       
              
        plt.plot(x, diff_distribution.pdf(x), color='r', lw=1.0)
        plt.fill_between(x, diff_distribution.pdf(x), y2=0, color='r', alpha=.2)
        plt.legend(['same', 'different'])
        plt.xlabel('cosine similarity')
        #plt.xlim([-1, 1])

        ax = plt.gca()
        ax.axes.yaxis.set_ticks([])
        plt.tight_layout(pad=0)
        plt.show()

        if self.mode == 'test':

            for set_size in range(1,5):
                #set_size = self.params['model_params']['set_size']
                scores, true = calc_scores(X, y, num_authors, set_size,
                                           log_odds_same_independent,
                                           same_distribution.pdf, diff_distribution.pdf)
                fpr, tpr, _ = roc_curve(true, scores)
                plt.figure(1, figsize=(2.5, 2.0), dpi=220)
                auc = sklearn.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, label=str(set_size) + '(AUC: {:.5f})'.format(auc), lw=1.0)

                plt.xlabel("FPR")
                plt.ylabel("TPR")
                #plt.xlim([0, .5])
                #plt.ylim([.5, 1])
                plt.xscale('log')
                plt.tight_layout(pad=0)


                plt.figure(2)
                prec, recall, _ = precision_recall_curve(true, scores)        
                ax = plt.gca()
                disp = PrecisionRecallDisplay(precision=prec, recall=recall)
                disp.plot(ax, name=str(set_size))
                predictions = np.zeros(scores.shape)
                predictions[scores > .5] = 1.0
                print(predictions.shape)
                print("RESULTS: ", np.sum(predictions == true) / (num_authors * 2))
            plt.figure(1)
            plt.legend(prop={'size': 6})
            plt.show()
        

    def output_hyperparameter_metrics(self, directory):
        pass


    def make_arg_parser(self):
        super().make_arg_parser()
        self.parser.add_argument("-mode")
        self.parser.add_argument("-second_combs", nargs='+', type=int)

    def get_args(self):

        exp_type, exp, combination = super().get_args()

        self.secondary_combs = self.args["second_combs"]
        self.mode = self.args["mode"]
        if self.mode is None:
            self.mode = "train"

        return exp_type, exp, combination


if __name__ == "__main__":
    TrainHistogramVerifier().execute()
