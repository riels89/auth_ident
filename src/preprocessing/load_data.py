import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import itertools
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))
from src import TRAIN_LEN, SL
from src.preprocessing.pair_authors import pair_authors

chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
chars_to_encode = list(chars_to_encode)
len_encoding = len(chars_to_encode)
chars_index = [i for i in range(len_encoding)]

char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)

binary_encoding_len = 8
max_code_length = 6000

# max = 102400
max_chars = 0
counter = 0

def make_csv():
    # ->contest_id/
    # --->username/
    # ------>problem_id/
    # -------->solution_id/
    # ---------->extracted/
    # ------------>contestant submitted files

    filepaths = []
    usernames = []

    max_chars = 0
    counter = 0
    for root, _, files in os.walk("data/raw/gcj"):
        for file in files:
            filepaths.append(os.path.join(root, file))
            usernames.append(root.split(SL)[2])

    filepaths = pd.DataFrame({"username": usernames, "filepath": filepaths})
    filepaths.to_csv("refrences/gcj.csv")


def file_stats(file):
    global max_chars, counter
    characters = tf.strings.length(file)
    print(characters)
    max_chars = max(max_chars, characters)
    counter += 1
    print(counter)
    return file

def tf_file_stats(file, username):
    tf.py_function(file_stats, [file], [tf.string])
    return file, username

def encode_to_one_hot(code_to_embed):
    # start = tf.timestamp(name=None)

    reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
    encoding = table.lookup(reshaped)
    encoding = tf.reshape(tf.squeeze(tf.one_hot(encoding, len_encoding)), (-1, len_encoding))

    code_length = tf.shape(encoding)[0]
    padding = [[0, max_code_length - code_length], [0, 0]]
    encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=0)

    # end = tf.timestamp(name=None)
    # tf.print("Embedding time: ", [end - start])

    return encoding


bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)
def encode_to_binary(code_to_embed):
    reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
    encoding = tf.cast(table.lookup(reshaped) + 1, tf.uint8)
    unpacked = tf.reshape(tf.math.floormod(tf.cast(encoding[:, None] // bits, tf.int32), 2),
                          shape=(-1, binary_encoding_len))

    # unpacked = tf.reshape(tf.bitwise.bitwise_and(encoding,
    #                       tf.broadcast_to(b, tf.shape(encoding))), shape=(-1, len_encoding))
    # tf.print("Unpacked shape: ", tf.shape(unpacked))
    # tf.print("Unpacked: ", unpacked)
    code_length = tf.shape(unpacked)[0]
    padding = [[0, max_code_length - code_length], [0, 0]]
    encoding = tf.pad(unpacked, padding, 'CONSTANT', constant_values=0)

    return encoding


def get_dataset(batch_size, seed=13):

    def encode(code_to_embed, username):
        return encode_to_one_hot(encoding), username

    def get_file(file):
        # start = tf.timestamp(name=None)

        parts = tf.strings.split(file, '/')
        output = (tf.io.read_file(file), parts[2])

        # end = tf.timestamp(name=None)
        # tf.print("Get file time: ", [end - start])

        return output

    files = pd.read_csv("refrences/gcj.csv")

    dataset = tf.data.Dataset.from_tensor_slices(files['filepath'])
    dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda code, label: tf.strings.length(code) <= max_code_length)
    dataset = dataset.map(encode, tf.data.experimental.AUTOTUNE)

    # dataset = dataset.map(tf_file_stats)

    dataset = dataset.shuffle(1024, seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def get_dataset_pairs(batch_size, binary_encoding=False, seed=13):

    if binary_encoding:
        len_encoding = binary_encoding_len
    else:
        len_encoding = len(chars_to_encode)

    def encode_binary(file1, file2, label):
        return encode_to_binary(file1), encode_to_binary(file2), label

    def encode_one_hot(file1, file2, label):
        return encode_to_one_hot(file1), encode_to_one_hot(file2), label

    def get_file(files, label):
        return (tf.io.read_file(files[0]), tf.io.read_file(files[1]), label)

    def truncate_files(file1, file2, label):
        # start = tf.timestamp(name=None)

        truncted1 = tf.strings.substr(file1, pos=0, len=tf.math.minimum(tf.strings.length(file1), max_code_length))
        truncted2 = tf.strings.substr(file2, pos=0, len=tf.math.minimum(tf.strings.length(file2), max_code_length))
        # end = tf.timestamp(name=None)
        # tf.print("Get file time: ", [end - start])

        return truncted1, truncted2, label

    def concat_files(file1, file2, label):
        return tf.squeeze(tf.concat([file1, tf.ones([1, len_encoding], dtype=file1.dtype), file2], axis=0)), label

    def set_shape(files, labels):
        files.set_shape((max_code_length * 2 + 1, len_encoding))
        return files, labels

    def create_dataset(pairs, labels):
        # TODO: Do speed test comparing one large map vs many smaller maps

        dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))
        dataset = dataset.shuffle(4096, seed)
        dataset = dataset.repeat()
        dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(truncate_files, tf.data.experimental.AUTOTUNE)
        if binary_encoding:
            dataset = dataset.map(encode_binary, tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(encode_one_hot, tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(concat_files, tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(set_shape, tf.data.experimental.AUTOTUNE)
        # dataset = dataset.map(tf_file_stats)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset

    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = load_paired_file_paths()

    train_dataset = create_dataset(train_pairs, train_labels)
    val_dataset = create_dataset(val_pairs, val_labels)

    return train_dataset, val_dataset

def load_paired_file_paths():
    train_pairs = np.load('data/paired_file_paths/train_pairs.npy', allow_pickle=True)
    train_labels = np.load('data/paired_file_paths/train_labels.npy', allow_pickle=True)
    val_pairs = np.load('data/paired_file_paths/val_pairs.npy', allow_pickle=True)
    val_labels = np.load('data/paired_file_paths/val_labels.npy', allow_pickle=True)
    test_pairs = np.load('data/paired_file_paths/test_pairs.npy', allow_pickle=True)
    test_labels = np.load('data/paired_file_paths/test_labels.npy', allow_pickle=True)
    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels

def create_and_save_dataset():
    pa = pair_authors("refrences/gcj.csv")
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = pa.generate_pairs()

    np.save('data/paired_file_paths/train_pairs.npy', train_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/train_labels.npy', train_labels, allow_pickle=True)
    np.save('data/paired_file_paths/val_pairs.npy', val_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/val_labels.npy', val_labels, allow_pickle=True)
    np.save('data/paired_file_paths/test_pairs.npy', test_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/test_labels.npy', test_labels, allow_pickle=True)

# print(load_paired_file_paths()[0].shape)
# dataset,_ = get_dataset_pairs(512, binary_encoding=True)
# i = 0
# for batch in dataset:
#     print(i)
#     i+=1
# pair_authors_fast(1e6)
# print("Embedding time: " + str(sum(embedding_times) / tf.shape(embedding_times)[0]))
# print("Get file time: " + str(sum(get_file_times) / tf.shape(get_file_times)[0]))
# print(embedding_times)
# make_csv()