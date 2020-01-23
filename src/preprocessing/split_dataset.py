import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import itertools
import math

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src import TRAIN_LEN, SL
from src.preprocessing.pair_authors import pair_authors
from src.preprocessing import load_data

class split_dataset:

    def __init__(self, max_code_length, batch_size, binary_encoding=False):

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

        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)

        self.max_code_length = max_code_length
        self.batch_size = batch_size

        self.bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)

        # max = 102400
        # max_chars = 0
        # counter = 0

    def encode_to_one_hot(self, code_to_embed):
        # start = tf.timestamp(name=None)

        reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
        encoding = self.table.lookup(reshaped)
        encoding = tf.reshape(tf.squeeze(tf.one_hot(encoding, self.len_encoding)), (-1, self.len_encoding))

        code_length = tf.shape(encoding)[0]
        padding = [[0, self.max_code_length - code_length], [0, 0]]
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=1)

        # end = tf.timestamp(name=None)
        # tf.print("Embedding time: ", [end - start])

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

    def create_dataset(self, pairs, labels):

        def encode_binary(files, label):
            files["input_1"] = self.encode_to_binary(files["input_1"])
            files["input_2"] = self.encode_to_binary(files["input_2"])
            return files, label

        def encode_one_hot(files, label):
            files["input_1"] = self.encode_to_one_hot(files["input_1"])
            files["input_2"] = self.encode_to_one_hot(files["input_2"])
            return files, label

        def get_file(files, label):
            files["input_1"] = tf.io.read_file(files["input_1"])
            files["input_2"] = tf.io.read_file(files["input_2"])
            return files, label

        def truncate_files(files, label):
            # start = tf.timestamp(name=None)

            files["input_1"] = tf.stack([self.start, tf.strings.substr(files["input_1"], pos=0,
                                         len=tf.math.minimum(tf.strings.length(files["input_1"]),
                                         self.max_code_length)),
                                         self.end], axis=0)
            files["input_2"] = tf.stack([self.start, tf.strings.substr(files["input_2"], pos=0,
                                         len=tf.math.minimum(tf.strings.length(files["input_2"]),
                                         self.max_code_length)),
                                         self.end], axis=0)
            # end = tf.timestamp(name=None)
            # tf.print("Get file time: ", [end - start])

            return files, label

        def set_shape(files, label):
            files["input_1"].set_shape((self.max_code_length + 2, self.len_encoding))
            files["input_2"].set_shape((self.max_code_length + 2, self.len_encoding))
            return files, label
             
        dataset = tf.data.Dataset.from_tensor_slices(({"input_1": pairs[:, 0], "input_2": pairs[:, 1]}, labels))
        
        dataset = dataset.shuffle(4096)
        dataset = dataset.repeat()
        
        dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(truncate_files, tf.data.experimental.AUTOTUNE)
        if self.binary_encoding:
            dataset = dataset.map(encode_binary, tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(encode_one_hot, tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(set_shape, tf.data.experimental.AUTOTUNE)

        # dataset = dataset.map(tf_file_stats)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    def get_dataset(self):
        train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_data.load_paired_file_paths()

        train_dataset = self.create_dataset(train_pairs, train_labels)
        val_dataset = self.create_dataset(val_pairs, val_labels)

        return train_dataset, val_dataset

    