import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import itertools
import math

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src import TRAIN_LEN, VAL_LEN, TEST_LEN, SL
from src.preprocessing.pair_authors import pair_authors
from src.preprocessing import load_data
from src.data_processing_expt import pairs_generator


class split_dataset:

    def __init__(self, max_code_length, batch_size, binary_encoding=False, flip_labels=False, language=None):
        print("\nIn INIT\n", flush=True)
        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        self.start = "<start>"
        self.end = "<end>"
        chars_to_encode = [self.start, self.end] + list(chars_to_encode)
        self.len_encoding = len(chars_to_encode)
        chars_index = [i for i in range(self.len_encoding)]
        self.binary_encoding_len = 8

        self.binary_encoding = binary_encoding
        if binary_encoding:
            self.len_encoding = self.binary_encoding_len

        self.language = language
        if language is None:
            self.language = "python"

        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)

        self.max_code_length = max_code_length
        self.batch_size = batch_size

        self.bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)

        self.flip_labels = flip_labels

    def encode_to_one_hot(self, code_to_embed):
        reshaped = tf.concat([[self.start], tf.strings.unicode_split(code_to_embed, 'UTF-8'), [self.end]], axis=0)
        encoding = self.table.lookup(reshaped)
        encoding = tf.reshape(tf.squeeze(tf.one_hot(encoding, self.len_encoding)), (-1, self.len_encoding))

        code_length = tf.shape(encoding)[0]
        padding = [[0, self.max_code_length + 2 - code_length], [0, 0]]
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=1)

        return encoding

    def encode_to_binary(self, code_to_embed):
        reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
        encoding = tf.cast(self.table.lookup(reshaped) + 1, tf.uint8)
        unpacked = tf.reshape(tf.math.floormod(tf.cast(encoding[:, None] // self.bits, tf.int32), 2),
                              shape=(-1, self.binary_encoding_len))

        code_length = tf.shape(unpacked)[0]
        padding = [[0, self.max_code_length - code_length], [0, 0]]
        encoding = tf.pad(unpacked, padding, 'CONSTANT', constant_values=1)

        return encoding

    def flip_labels(self, files, label):
        if label == 0:
            label = 1
        else:
            label = 1
        return files, label

    def create_dataset(self, language, split):

        def encode_binary(dataset):
            dataset[0:,] = self.encode_to_binary(dataset[0:,])
            dataset[1:,] = self.encode_to_binary(dataset[1:,])

        #def encode_one_hot(dataset):
        #    dataset[0:,] = self.encode_to_one_hot(dataset[0:,])
        #    dataset[1:,] = self.encode_to_one_hot(dataset[1:,])

        def encode_one_hot(files, label):
            files["input_1"] = self.encode_to_one_hot(files["input_1"])
            files["input_2"] = self.encode_to_one_hot(files["input_2"])
            return files, label

        def set_shape(files, label):
            files["input_1"].set_shape((self.max_code_length + 2, self.len_encoding))
            files["input_2"].set_shape((self.max_code_length + 2, self.len_encoding))
            label = label
            return files, label

        if split == 'train':
            num_samples = TRAIN_LEN
        elif split == 'val':
            num_samples = VAL_LEN
        elif split == 'test':
            num_samples = TEST_LEN
        else:
            print("ERROR: Invalid split type in split_dataset.create_dataset: " + split)
            exit(1)

        file="data/loaded/" + language + "_" + split + ".h5"
        df = pd.read_hdf(file)
        pg = pairs_generator.PairGen(df, crop_length=self.max_code_length, samples_per_epoch=num_samples)


        print("Generating Data...", flush=True)
        #data = np.array(list(pg.gen()))
        #print("Data Generated.\nLoading Data...", flush=True)
        #dataset = tf.data.Dataset.from(({"input_1": data[:, 0], "input_2": data[:, 1]}, data[:,2].astype(int)))
        dataset = tf.data.Dataset.from_generator(
            pg.gen,
            ({"input_1": tf.string, "input_2": tf.string}, tf.bool),
            output_shapes=({"input_1": tf.TensorShape([]),
                            "input_2": tf.TensorShape([])},
                           tf.TensorShape([])))

        print("Data Generated.", flush=True)

        #dataset = dataset.shuffle(4096)
        dataset = dataset.map(encode_one_hot)
        #dataset = dataset.repeat()

        if self.binary_encoding:
            print("ERROR: Binary encoding not supported: split_dataset.create_dataset")
            exit(1)
        if self.flip_labels:
            print("ERROR: Flip Labels not supported: split_dataset.create_dataset")
            exit(1)

        dataset = dataset.map(set_shape, 120)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    def get_dataset(self):
        print("\nIn get_dataset()\n", flush=True)
        train_dataset = self.create_dataset(self.language, "train")
        val_dataset = self.create_dataset(self.language, "val")
        test_dataset = self.create_dataset(self.language, "test")

        return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    sds = SplitDataset(20, 4, language='java')
    train_dataset, val_dataset, test_dataset = sds.get_dataset()
    print(list(train_dataset.take(3).as_numpy_iterator()))
