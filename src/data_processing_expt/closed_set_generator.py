"""
Code for creating on-the-fly random file pairings.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import itertools


class ClosedGen:
    def __init__(self, dataframe, crop_length, match_rate=.5,
                 samples_per_epoch=1000, k_cross_val=5):
        if (k_cross_val < 2):
            print("k_cross_val ust be greater than 1.")
            exit(1)

        self.dataframe = dataframe
        self.crop_length = crop_length
        self.match_rate = match_rate
        self.samples_per_epoch = samples_per_epoch
        self.k_cross_val = k_cross_val

        self.rng = np.random.default_rng(1)

        self.num_files = len(dataframe)

        #For one-hot
        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        self.start = "<start>"
        self.end = "<end>"
        chars_to_encode = [self.start, self.end] + list(chars_to_encode)
        self.len_encoding = len(chars_to_encode)
        chars_index = [i for i in range(self.len_encoding)]
        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)


        # Mapping from author names to file index
        # ["tom"] -> [1, 17, 37]
        self.files_by_auth_name = dataframe.groupby(['username']).indices

        # reverse file to author mapping
        # [17] -> ["tom"]
        # [37] -> ["tom"]
        self.indx_to_auth = {}
        for item in self.files_by_auth_name.items():
            for indx in item[1]:
                self.indx_to_auth[indx] = item[0]

        # Store all of the files by authors with more than k files.
        # [[4,27,31,45,58],[...]]
        self.files_with_k = np.array(list(itertools.chain.from_iterable(
            filter(
                lambda x:
                len(x) >= self.k_cross_val,
                self.files_by_auth_name.values()))))

#    def get_splits(self):
        #for i in range(20):
        #    num_auth = len(list(itertools.chain.from_iterable(filter(lambda x: len(x) > i, self.files_by_auth_name.values()))))
        #    print("num files: ", i, "num_auth:", num_auth)

        #return train_gen, other_gen

    def gen(self):
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

        """
        X = []
        y = []
        authors_seen = []
        for _ in range(self.samples_per_epoch):
            #TODO Change to each author being equally likely?
            loop_test = True
            while loop_test:
                rand_file = self.rng.choice(self.files_with_k, 1,
                                                shuffle=False)[0]
                rand_auth = self.indx_to_auth[rand_file]
                loop_test = len(self.files_by_auth_name[rand_auth]) < self.k_cross_val

            rand_k = self.rng.choice(self.files_by_auth_name[rand_auth],
                                            self.k_cross_val, replace=False, shuffle=False)

            if authors_seen.__contains__(rand_auth):
                auth_indx = authors_seen.index(rand_auth)
            else:
                auth_indx = len(authors_seen)
                authors_seen.append(rand_auth)

            for i in rand_k:
                X.append(self.encode_to_one_hot(self.random_crop(i, self.crop_length)))
            #X.extend(rand_k)
            y.extend(self.k_cross_val * [auth_indx])

        return X, y

    def random_crop(self, file_indx, crop_length):
        """
        Return a random crop from the file at the provided index. If
        crop_length is longer than the length of the file, then the entire
        file will be returned.
        """
        contents = self.dataframe['file_content'][file_indx]
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
    pg = ClosedGen(df, crop_length=100, samples_per_epoch=15)
    X,y = pg.gen()
    print(X)
    print(y)
    exit()
    for i in X:
        print(i)

    for i in y:
        print(i)

