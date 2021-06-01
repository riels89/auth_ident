from auth_ident.models import GenericSecondaryClassifier 
from auth_ident.datasets import ClosedDataset

from sklearn.model_selection import cross_val_score, train_test_split, KFold

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted. 

    https://stackoverflow.com/a/56175538
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def log_odds_same_averaged(points_a, points_b, same_pdf, diff_pdf, prior_same):

    
    dists = -cdist(points_a, points_b, metric='cosine') + 1
    rolls = np.arange(0, -points_a.shape[0], -1)
    rolled_dists = indep_roll(dists, rolls)  # align all diagonals into columns

    small = np.ones(rolled_dists.shape) * .00000000001
    sames = np.maximum(same_pdf(rolled_dists), small)
    diffs = np.maximum(diff_pdf(rolled_dists), small)

    accs = np.ones(diffs.shape[1]) *  np.log(prior_same / (1-prior_same))

    accs += np.sum(np.log(sames / diffs), axis=0)

    # It turns out that averaging the log odds then converting to probability
    # gives a different answer than converting to probabilities, then averaging
    # Averaging the probabilities makes more sense.
    #print(np.mean(np.exp(accs) / (1 + np.exp(accs))))
    
    return np.mean(np.exp(accs) / (1 + np.exp(accs)))

def log_odds_same_independent(points_a, points_b, same_pdf, diff_pdf, prior_same):
    dists = -cdist(points_a, points_b, metric='cosine').flatten() + 1
    
    small = np.ones(dists.shape) * .00000000001
    sames = np.maximum(same_pdf(dists), small)
    diffs = np.maximum(diff_pdf(dists), small)

    acc = np.log(prior_same / (1-prior_same))

    
    acc += np.sum(np.log(sames / diffs))

    #print(np.exp(acc) / (1 + np.exp(acc)))
    
    return acc

def calc_scores(X, y, num_authors, set_size, comparison_func,
                same_pdf, diff_pdf):
    """author labels in y must sequentially repeat every num_author
    entires.

    """
    
    scores = np.empty(num_authors * 2)
    true = np.zeros(num_authors * 2)
    true[0:num_authors] = 1

    # same author calculations
    for i in range(num_authors):
        indices_a = np.arange(i, num_authors * set_size, num_authors)
        indices_b = np.arange(i + set_size * num_authors,
                              (set_size * 2) * num_authors,
                              num_authors)
        scores[i] = comparison_func(X[indices_a,...], X[indices_b,...],
                                   same_pdf, diff_pdf, .5)
        
        
    for i in range(num_authors):
        indices_a = np.arange(i, num_authors * set_size, num_authors)
        indices_b = np.arange(i + set_size * num_authors,
                              (set_size * 2) * num_authors,
                              num_authors)
        indices_b = (indices_b + 1) % X.shape[0] # shift by one to ensure non-match
        scores[num_authors + i] = comparison_func(X[indices_a,...], X[indices_b,...],
                                                  same_pdf, diff_pdf, .5)
    return scores, true
        

def create_pdfs(points, labels):
    same_dists = []
    for label in np.unique(labels):
        i_points = points[np.argwhere(labels == label).flatten(), ...]
        cur_dists = -cdist(i_points, i_points, metric='cosine') + 1
        tril = np.tril_indices(cur_dists.shape[0], -1)
        same_dists.append(cur_dists[tril].flatten())
    all_same_dists = np.asarray(same_dists).flatten()

    diff_dists = []
    num_points_per_class = int(points.shape[0] / np.unique(labels).size)
    for i in range(num_points_per_class):
        i_indices = np.arange(i, points.shape[0], num_points_per_class)
        i_points = points[i_indices, ...]
        cur_dists = -cdist(i_points, i_points, metric='cosine') + 1
        tril = np.tril_indices(cur_dists.shape[0], -1)
        diff_dists.append(cur_dists[tril])
    all_diff_dists = np.asarray(diff_dists).flatten()

    diff_hist = np.histogram(all_diff_dists, bins='auto')
    same_hist = np.histogram(all_same_dists, bins='auto')

    diff_distribution = scipy.stats.rv_histogram(diff_hist)
    same_distribution = scipy.stats.rv_histogram(same_hist)

    return same_distribution, diff_distribution

class HistogramVerifier(GenericSecondaryClassifier):
    """

    """
    def __init__(self, params, combination, logger, logdir):
        super().__init__(params, combination, logger, logdir)

        self.name = "histogram_verifier"
        self.dataset = ClosedDataset

        self.model = self.params["model_params"]

    def train(self, X, y):

        assert(X.shape[0] == 1600 * 9)
        print(self.model)
        X_train = X[0:1600*1 ,...]
        X_test = X[1600*1::, ...]
        y_train = y[0:1600*1]
        y_test = y[1600*1::]


        import pickle
        if False:
            # Hack... Manually set things up to create the distributions using
            # the validation split, then change the flag to use
            # distribution using the test split.
            same_distribution, diff_distribution = create_pdfs(X, y)
            pickle.dump(same_distribution, open("same_dist.pkl", "wb" ))
            pickle.dump(diff_distribution, open("diff_dist.pkl", "wb" ))
            return
        else:
            same_distribution = pickle.load(open("same_dist.pkl", "rb"))
            diff_distribution = pickle.load(open("diff_dist.pkl", "rb"))
            
        # PLOT THE HISTOGRAMS
        x = np.linspace(same_distribution.support()[0],
                        same_distribution.support()[1], 100, endpoint=False)
      
        plt.plot(x, same_distribution.pdf(x), color='b')
        plt.fill_between(x, same_distribution.pdf(x), y2=0, color='b', alpha=.2)

        x = np.linspace(diff_distribution.support()[0],
                        diff_distribution.support()[1], 100, endpoint=False)
        
       
              
        plt.plot(x, diff_distribution.pdf(x), color='r')
        plt.fill_between(x, diff_distribution.pdf(x), y2=0, color='r', alpha=.2)
        plt.legend(['same author', 'different author'])
        plt.xlabel('cosine similarity')
        #plt.xlim([-1, 1])
        plt.show()

        for set_size in range(1,5):
            #set_size = self.params['model_params']['set_size']
            scores, true = calc_scores(X, y, 1600, set_size,
                                       log_odds_same_averaged,
                                       same_distribution.pdf, diff_distribution.pdf)
            fpr, tpr, _ = roc_curve(true, scores)
            plt.figure(1)
            plt.plot(fpr, tpr, label=str(set_size))
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.xlim([0, .5])
            plt.ylim([.5, 1])

            
            plt.figure(2)
            prec, recall, _ = precision_recall_curve(true, scores)        
            ax = plt.gca()
            disp = PrecisionRecallDisplay(precision=prec, recall=recall)
            disp.plot(ax, name=str(set_size))
            predictions = np.zeros(scores.shape)
            predictions[scores > .5] = 1.0
            print(predictions.shape)
            print("RESULTS: ", np.sum(predictions == true) / (1600 * 2))
        plt.figure(1)
        plt.legend()
        plt.show()
        



        return np.sum(predictions == true) / (1600 * 2)

    def evaluate(self, X, y=None): 

        return -1
