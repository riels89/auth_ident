import numpy as np
from scipy.spatial.distance import cdist
import scipy
import sklearn.metrics

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
    
    return np.exp(acc) / (1 + np.exp(acc))

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
