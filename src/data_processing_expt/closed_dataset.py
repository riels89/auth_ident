"""
Code for creating on-the-fly random file pairings.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
from time import perf_counter


class closed_dataset:
    def __init__(self, crop_length, k_cross_val=5, language="python"):
        if (k_cross_val < 2):
            print("k_cross_val ust be greater than 1.")
            exit(1)

        self.language = language
        self.crop_length = crop_length
        self.k_cross_val = k_cross_val

        self.rng = np.random.default_rng(1)

        #For one-hot
        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        self.start = "<start>"
        self.end = "<end>"
        chars_to_encode = [self.start, self.end] + list(chars_to_encode)
        self.len_encoding = len(chars_to_encode)
        chars_index = [i for i in range(self.len_encoding)]
        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)

        # Load dataframes
        file = "data/loaded/" + self.language + "_val.h5"
        self.dataframe1 = pd.read_hdf(file)
        file = "data/loaded/" + self.language + "_test.h5"
        self.dataframe2 = pd.read_hdf(file)

    def get_datasets(self):
        return self.get_two(self.dataframe1), self.get_two(self.dataframe2)

    def get_two(self, df):
        """
        Generate file pairings where each file is equally likely to be
        included in a pairing.

         Algorithm:
        Take all authors with >= k files
        while num_files < files_requested:
            pick i randomly from authors(with atleast k files remaining)
                crop, encode, and add k files from i to the dataset
                if i in authors_seen:
                    label = authors_seen.index(author)
                else:
                    label = len(authors_seen)
                    authors_seen.append(i)

        Algorithm 2.0:
        files = k files from all authors with >= k files
        for i in len(files)
            y[i] = author(files[i])
        """

        # TODO Make faster

        # Mapping from author names to file index
        # ["tom"] -> [1, 17, 37]
        self.files_by_auth_name = df.groupby(['username']).indices

        # Map from authors with >= k files to their files (indx)
        #  ["Larry"] -> [4, 34, 67, 231, 453, 768]
        self.authors_with_k = dict(filter(lambda x: len(x[1]) >= self.k_cross_val, self.files_by_auth_name.items()))

        # Modifies the map s.t. each author has exactly k files
        for k in self.authors_with_k:
            self.authors_with_k[k] = self.rng.choice(self.authors_with_k[k], self.k_cross_val, replace=False,
                                                     shuffle=False)


        # List of all files sorted by author where each author has exactly k files
        files = np.concatenate(list(self.authors_with_k.values()))

        # Generate labels
        y = []
        for i in range(len(files)):
            y.append(int(i / self.k_cross_val))

        crop = np.vectorize(self.random_crop)
        files = crop(files, self.crop_length)

        X = []
        for i in range(len(files)):
            X.append(self.encode_to_one_hot(files[i]))
        X = np.array(X)

        return X, y

    def random_crop(self, file_indx, crop_length, df):
        """
        Return a random crop from the file at the provided index. If
        crop_length is longer than the length of the file, then the entire
        file will be returned.
        """
        contents = self.df['file_content'][file_indx]
        if len(contents) > crop_length:
            start = self.rng.integers(0, len(contents) - crop_length + 1)
            contents = contents[start:start + crop_length]
        return contents.ljust(crop_length, '\0')

    def encode_to_one_hot(self, code_to_embed):
        reshaped = tf.concat([[self.start], tf.strings.unicode_split(code_to_embed, 'UTF-8'), [self.end]], axis=0)
        encoding = self.table.lookup(reshaped)
        encoding = tf.reshape(tf.squeeze(tf.one_hot(encoding, self.len_encoding)), (-1, self.len_encoding))

        code_length = tf.shape(encoding)[0]
        padding = [[0, self.crop_length + 2 - code_length], [0, 0]]
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=1)
        return encoding

if __name__ == "__main__":
    df = pd.read_hdf('data/loaded/cpp_test.h5')
    pg = closed_dataset(df, crop_length=1200)
    #X, y = pg.gen()

