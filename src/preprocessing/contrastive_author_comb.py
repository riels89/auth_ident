import pandas as pd
import numpy as np
import itertools

class contrstive_author_comb():
    """Generates pairs of authors. Author pairs will be pairs of the same author,
    based on the match rate percentage. The rest of the time Author pairs will be
    of different authors."""

    def __init__(self, filepath, author_val_test_split=0.1, train_files_per_auth=20, val_files_per_auth=20):

        self.train_files_per_auth = train_files_per_auth
        self.val_files_per_auth = val_files_per_auth

        self.author_val_test_split = author_val_test_split
        self.author_train_split = 1 - 2 * author_val_test_split

        self.files = pd.read_csv(filepath, keep_default_na=False)
        self.authors = list(set(self.files['username']))
        self.auth_to_idx = {self.authors[i]: i for i in range(len(self.authors))}
        self.idx_to_auth = {v: k for k, v in self.auth_to_idx.items()}

        self.authors = np.array([self.auth_to_idx[author] for author in self.authors])

        self.files_by_auth = self.files.groupby(['username']).indices # dict
        self.files_by_auth = np.array([np.array(self.files_by_auth[self.idx_to_auth[auth_idx]]) for auth_idx in self.authors])

    def generate_pairs(self):

        shuffle_mask = np.random.permutation(len(self.authors))

        train_auth_len = int(self.author_train_split * self.authors.shape[0])
        val_test_auth_len = int(self.author_val_test_split * self.authors.shape[0])

        train_mask = shuffle_mask[:train_auth_len]
        val_mask = shuffle_mask[train_auth_len: train_auth_len + val_test_auth_len]
        test_mask = shuffle_mask[train_auth_len + val_test_auth_len: train_auth_len + 2 * val_test_auth_len]

        train_authors = self.authors[train_mask]
        val_authors = self.authors[val_mask]
        test_authors = self.authors[test_mask]

        train_pairs = self.__pair(train_authors, self.train_samples, self.train_files_per_auth)
        val_pairs = self.__pair(val_authors, self.val_test_samples, self.val_files_per_auth)
        test_pairs = self.__pair(test_authors, self.val_test_samples, self.val_files_per_auth)

        return train_pairs, val_pairs, test_pairs


    def __pair(self, authors, files_per_auth):

        author_names = np.array([self.idx_to_auth[author] for author in authors])
        author_file_counts = self.files[self.files['username'].isin(author_names)]['username'].value_counts()

        # Get pairs of the same author
        authors_to_match = self.match_authors(author_file_counts, files_per_auth)
        file_h_c = self.get_all_file_combinations(author_file_counts)
        matched_combinations = self.select_matched_files(authors_to_match, file_h_c)
        print("Same author pairing shape: " + str(matched_combinations.shape))

        shuffle_mask = np.random.permutation(matched_combinations.shape[0])
        data = matched_combinations[shuffle_mask]
        data = self.files['filepath'].take(data.flatten()).values.reshape(-1, 2)

        return data

    def match_authors(self, author_file_counts, files_per_auth):
        """Gets number of times each author will match with themselvs"""
        authors_to_match = []
        for index in range(2, files_per_auth):
            # authors hat refers to authors with a file count > index
            authors_h_i_name = author_file_counts[author_file_counts > index].index
            authors_h_i = [self.auth_to_idx[author] for author in authors_h_i_name]

            authors_to_match.append(authors_h_i)
        return authors_to_match

    def get_all_file_combinations(self, author_file_counts):
        # authors hat refers to authors with a file count > 1
        authors_h_name = author_file_counts[author_file_counts > 1].index
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
        authors_to_match = np.sort(np.concatenate(authors_to_match))
        authors_to_match_set, set_inverse = np.unique(authors_to_match, return_inverse=True)
        auth_bin_cnt = np.bincount(set_inverse)

        combinations_idx = []
        for author in range(len(auth_bin_cnt)):
            combinations_idx.append(np.random.choice(len(file_h_c[author]), auth_bin_cnt[author], replace=False))

        # Convert to numpy array
        matched_combinations = np.array([np.array(file_h_c[author][idx])
                                        for author in range(len(combinations_idx))
                                        for idx in combinations_idx[author]])
        return matched_combinations
        