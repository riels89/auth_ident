import pandas as pd
import numpy as np
import itertools
from scipy.special import comb 


class contrastive_author_comb():
    """Generates pairs of authors. Author pairs will be pairs of the same author,
    based on the match rate percentage. The rest of the time Author pairs will be
    of different authors."""

    def __init__(self, filepath, author_val_test_split=0.1, 
                 train_matches_per_auth=200, val_matches_per_auth=200):

        self.train_matches_per_auth = train_matches_per_auth
        self.val_matches_per_auth = val_matches_per_auth

        self.author_val_test_split = author_val_test_split
        self.author_train_split = 1 - 2 * author_val_test_split

        self.files = pd.read_csv(filepath, keep_default_na=False)
        self.authors = list(set(self.files['username']))
        self.auth_to_idx = {self.authors[i]: i for i in range(len(self.authors))}
        self.idx_to_auth = {v: k for k, v in self.auth_to_idx.items()}

        # Transform authors to their index
        self.authors = np.array([self.auth_to_idx[author] for author in self.authors])

        self.files_by_auth = self.files.groupby(['username']).indices # dict
        # Groups authors files by the author index
        # (idx, np tuple of files)
        self.files_by_auth = np.array([
            np.array(self.files_by_auth[self.idx_to_auth[auth_idx]])
            for auth_idx in self.authors])

    def generate_pairs(self):
        
        # Gets random author permuation of idxs 
        shuffle_mask = np.random.permutation(len(self.authors))

        # Get number of authors for train, val, and test. Test size = val size
        train_auth_len = int(self.author_train_split * self.authors.shape[0])
        val_test_auth_len = int(self.author_val_test_split * self.authors.shape[0])

        # Grab the correct number of random authors
        train_mask = shuffle_mask[:train_auth_len]
        val_mask = shuffle_mask[train_auth_len: train_auth_len + val_test_auth_len]
        test_mask = shuffle_mask[train_auth_len + val_test_auth_len: train_auth_len + 2 * val_test_auth_len]

        # Asign the authors
        train_authors = self.authors[train_mask]
        val_authors = self.authors[val_mask]
        test_authors = self.authors[test_mask]

        # Pair the similar authors
        train_pairs = self.__pair(train_authors, self.train_matches_per_auth)
        val_pairs = self.__pair(val_authors, self.val_matches_per_auth)
        test_pairs = self.__pair(test_authors, self.val_matches_per_auth)

        return train_pairs, val_pairs, test_pairs


    def __pair(self, authors, matches_per_author):

        # Get the author names from the indexs to use to get the file counts
        # per authors
        author_names = np.array([self.idx_to_auth[author] for author in authors])
        author_file_counts = self.files[self.files['username'].isin(author_names)]['username'].value_counts()

        # Get pairs of the same author
        authors_to_match = self.match_authors(author_file_counts, matches_per_author)
        file_h_c = self.get_all_file_combinations(author_file_counts)
        matched_combinations = self.select_matched_files(authors_to_match, file_h_c)
        print("Same author pairing shape: " + str(matched_combinations.shape))

        shuffle_mask = np.random.permutation(matched_combinations.shape[0])
        data = matched_combinations[shuffle_mask]
        data = self.files['filepath'].take(data.flatten()).values.reshape(-1, 2)

        return data

    def match_authors(self, author_file_counts, matches_per_author):
        """Gets number of times each author will match with themselvs"""
        authors_to_match = []

        combinations = {i: np.minimum(matches_per_author, int(comb(i, 2, False)))
                        for i in range(2, matches_per_author + 1)} 
        # Get all authors with file counts >= 2
        authors_with_min = author_file_counts[author_file_counts >= 2]
        # Gets the max matches per author or max number of files
        authors_with_min = authors_with_min.apply(
            lambda count: combinations[np.minimum(matches_per_author, count)])
        # Order the counts based on global author index. Dicts in python
        # > 3.6 are insertion ordered so the indexes will stay in order
        authors_to_match = np.array([authors_with_min[author] 
                                     for author, idx in self.auth_to_idx.items() 
                                     if author in authors_with_min.index])

        return authors_to_match

    def get_all_file_combinations(self, author_file_counts):
        # authors hat refers to authors with a file count >= 2 
        authors_h_name = author_file_counts[author_file_counts >= 2].index
        authors_h = np.sort([self.auth_to_idx[author] for author in authors_h_name])

        print("num authors: " + str(len(author_file_counts)))
        print("num authors file count > 1: " + str(len(authors_h)))

        # Grouping the files by author
        file_h = self.files.groupby(['username']).indices # dict
        file_h = np.array([file_h[self.idx_to_auth[auth_idx]] for auth_idx in authors_h])

        # Getting all possible combinations
        file_h_c = np.array([np.array(list(itertools.combinations(file_set, 2)), dtype=(int, int))
                            for file_set in file_h])
        return file_h_c

    def select_matched_files(self, authors_to_match, file_h_c):
        """Maps the given authors to a random combination of files, without duplicates"""
        # Bin duplicate authors, the bin index is replaced by their count.
        # authors_to_match:=[1, 2, 2, 2, 3, 3] => [0, 1, 1, 1, 2, 2] => [1, 3, 2] = auth_bin_cnt
        # authors_to_match = np.sort(np.concatenate(authors_to_match))
        # authors_to_match_set, set_inverse = np.unique(authors_to_match, return_inverse=True)
        # auth_bin_cnt = np.bincount(set_inverse)
        total_comb = 0
        combinations_idx = []
        for author in range(len(authors_to_match)):
            total_comb = total_comb + authors_to_match[author]
            combinations_idx.append(np.random.choice(len(file_h_c[author]),
                                                     authors_to_match[author],
                                                     replace=False))

        lengths = [len(i) for i in combinations_idx]
        print("Median matches per author: ", np.median(lengths))
        print("Average: ", total_comb / float(len(authors_to_match)))
        
        # Convert to numpy array
        matched_combinations = np.array([np.array(file_h_c[author][idx])
                                        for author in range(len(combinations_idx))
                                        for idx in combinations_idx[author]])
        return matched_combinations
        
