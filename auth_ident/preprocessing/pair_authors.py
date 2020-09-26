import pandas as pd
import numpy as np
import itertools

class PairAuthors():
    """Generates pairs of authors. Author pairs will be pairs of the same author,
    based on the match rate percentage. The rest of the time Author pairs will be
    of different authors."""

    def __init__(self, filepath, match_rate=0.5, author_val_test_split=0.1, train_samples=1000000, val_test_samples=50000):
        self.total_samples = train_samples + 2 * val_test_samples
        self.train_samples = train_samples
        self.val_test_samples = val_test_samples
        self.match_rate = match_rate

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
        self.files_by_auth = np.array([np.array(self.files_by_auth[self.idx_to_auth[auth_idx]]) for auth_idx in self.authors])

    def generate_pairs(self):

        # Gets random author permuation of idxs 
        shuffle_mask = np.random.permutation(len(self.authors))

        # Get number of authors for train, val, and test Test size = val size
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

        # Pair the authors
        train_pairs, train_labels = self.__pair(train_authors, self.train_samples)
        val_pairs, val_labels = self.__pair(val_authors, self.val_test_samples)
        test_pairs, test_labels = self.__pair(test_authors, self.val_test_samples)

        return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels


    def __pair(self, authors, num_samples):

        # Get the author names from the indexs to use to get the file counts
        # per authors
        author_names = np.array([self.idx_to_auth[author] for author in authors])
        author_file_counts = self.files[self.files['username'].isin(author_names)]['username'].value_counts()

        # Get the number of samples for matched and nonmatched samples
        non_matched_length = int(num_samples * (1 - self.match_rate))
        matched_length = int(num_samples * self.match_rate)

        # Get pairs of the same author
        authors_to_match = self.match_authors(author_file_counts, matched_length)
        file_h_c = self.get_all_file_combinations(author_file_counts)
        matched_combinations = self.select_matched_files(authors_to_match, file_h_c)
        print("Same author pairing shape: " + str(matched_combinations.shape))

        # Get pairs of different authros
        first_author, second_author = self.select_authors(authors, non_matched_length)
        first_files = self.choose_files(first_author, non_matched_length)
        second_files = self.choose_files(second_author, non_matched_length)
        non_matched_combinations = np.column_stack([first_files, second_files])
        print("Different author pairings shape: ", non_matched_combinations.shape)

        # Combine combinations
        data = np.concatenate([matched_combinations, non_matched_combinations])
        labels = np.concatenate([np.ones(matched_combinations.shape[0]),
                                 np.zeros(non_matched_combinations.shape[0])])

        # Shuffle and resahpe data
        shuffle_mask = np.random.permutation(data.shape[0])
        data = data[shuffle_mask]
        data = self.files['filepath'].take(data.flatten()).values.reshape(-1, 2)
        labels = np.squeeze(labels[shuffle_mask])

        return data, labels

    def match_authors(self, author_file_counts, num_samples):
        """Gets number of times each author will match with themselvs"""
        authors_to_match = []
        samples_left = num_samples
        index = 1
        while(samples_left > 0):
            
            # Gets author names with file count > index
            authors_with_min = author_file_counts[author_file_counts > index].index
            # Transforms author names to idx
            authors_with_min_idx = [self.auth_to_idx[author] for author in authors_with_min]
            
            # If number of samples left is more than number of samples
            # to add, then just append otherwise grab a random subset
            # the size of the remaining samples
            if samples_left > len(authors_with_min_idx):
                authors_to_match.append(authors_with_min_idx)
            else:
                authors_to_match.append(np.random.choice(authors_with_min_idx,
                                                         samples_left,
                                                         replace=False))

            samples_left -= len(authors_with_min_idx)
            index += 1
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
        authors_to_match_set, set_inverse = np.unique(authors_to_match,
                                                      return_inverse=True)
        auth_bin_cnt = np.bincount(set_inverse)

        combinations_idx = []
        for author in range(len(auth_bin_cnt)):
            print("Number of files", len(file_h_c[author]))
            print("Files requested", auth_bin_cnt[author])
            combinations_idx.append(np.random.choice(len(file_h_c[author]),
                                                     auth_bin_cnt[author],
                                                     replace=False))

        # Convert to numpy array
        matched_combinations = np.array([np.array(file_h_c[author][idx])
                                        for author in range(len(combinations_idx))
                                        for idx in combinations_idx[author]])
        return matched_combinations

    def select_authors(self, authors, num_samples):
        """Select a random author, then select random author indices from num authors - 1 because
         we don't want to count the same author twice. Then for each second author index
         that is >= the first author, add one. This ensures we select from the correct
         set of possible values and avoids selecting the same author.
         Ex: The first authors to choose from is:  [1, 2, 3, 4, 5]
         The second authors to choose from is: [1, 2, 3, 4]
         If the first chosen author is 2
         2 >= 2 so 2 => 3. 3 => 4.  4 => 5
         Then the second author actually chooses from: [1, 3, 4, 5]"""
        first_author = np.squeeze(np.random.choice(authors.shape[0], [num_samples, 1]))
        second_author = np.squeeze(np.random.choice(authors.shape[0] - 1, [num_samples, 1]))
        bool_mask = second_author >= first_author
        second_author += bool_mask
        first_author = authors[first_author]
        second_author = authors[second_author]
        return first_author, second_author

    def choose_files(self, authors, non_matched_length):
        """Maps a set of authors to a set of randomly chosen files"""
        files_len = np.array([self.files_by_auth[author].shape[0] for author in authors])
        chosen_file = (np.random.rand(*authors.shape) * np.squeeze(files_len)).astype(int)
        files = self.files_by_auth[authors]
        assert authors.shape[0] == non_matched_length
        files = np.array([files[i][chosen_file[i]] for i in range(authors.shape[0])])
        return files
        
