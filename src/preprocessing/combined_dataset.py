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

class combined_dataset:

    def __init__(self, max_code_length, batch_size, binary_encoding=False):

        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        chars_to_encode = list(chars_to_encode)
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
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=0)

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
        encoding = tf.pad(unpacked, padding, 'CONSTANT', constant_values=0)

        return encoding

    def create_dataset(self, pairs, labels):

        def encode_binary(file1, file2, label):
            return self.encode_to_binary(file1), self.encode_to_binary(file2), label

        def encode_one_hot(file1, file2, label):
            return self.encode_to_one_hot(file1), self.encode_to_one_hot(file2), label

        def get_file(files, label):
            return (tf.io.read_file(files[0]), tf.io.read_file(files[1]), label)

        def truncate_files(file1, file2, label):
            # start = tf.timestamp(name=None)

            truncted1 = tf.strings.substr(file1, pos=0, len=tf.math.minimum(tf.strings.length(file1), self.max_code_length))
            truncted2 = tf.strings.substr(file2, pos=0, len=tf.math.minimum(tf.strings.length(file2), self.max_code_length))
            # end = tf.timestamp(name=None)
            # tf.print("Get file time: ", [end - start])

            return truncted1, truncted2, label

        def concat_files(file1, file2, label):
            return tf.squeeze(tf.concat([file1, tf.ones([1, self.len_encoding], dtype=file1.dtype), file2], axis=0)), label
                
        def set_shape(files, label):
            files.set_shape((self.max_code_length * 2 + 1, self.len_encoding))
            return files, label
             
        dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))
        
        dataset = dataset.shuffle(4096)
        dataset = dataset.repeat()
        dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(truncate_files, tf.data.experimental.AUTOTUNE)
        if self.binary_encoding:
            dataset = dataset.map(encode_binary, tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(encode_one_hot, tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(concat_files, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(set_shape, tf.data.experimental.AUTOTUNE)

        # dataset = dataset.map(tf_file_stats)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)

        return dataset

    def get_dataset(self):
        train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_data.load_paired_file_paths()

        train_dataset = self.create_dataset(train_pairs, train_labels)
        val_dataset = self.create_dataset(val_pairs, val_labels)

        return train_dataset, val_dataset

      