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
# from src.preprocessing.contrastive_author_comb import pair_authors as contrastive_pair

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
            filepaths.append(os.path.join(root, file).replace("\\", "/"))
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

def create_file_csv():
    pa = contrastive_pair("refrences/gcj.csv")
    train_pairs, val_pairs, test_pairs = pa.generate_pairs()
    files = pd.read_csv("refrences/gcj.csv", keep_default_na=False)
    authors = list(set(files['username']))
    auth_to_idx = {authors[i]: i for i in range(len(authors))}

    def make_file_csv(pair_set):
        files = np.empty([pair_set.shape[0], 4], dtype='object, object, i8, i8')

        for i in range(pair_set.shape[0]):
            with open(pair_set[i][0], 'r') as file:
                code = file.read()
                files[i, 0] = code
            with open(pair_set[i][1], 'r') as file:
                code = file.read()
                files[i, 1] = code
            files[i, 2] = auth_to_idx[pair_set[i][0].split(SL)[2]]
            files[i, 3] = auth_to_idx[pair_set[i][1].split(SL)[2]]

        return pd.DataFrame(data=files, columns=["file1, file2, author1", "author2"])

    train_df = make_file_csv(train_pairs)
    train_df.to_csv('data/paired_file_paths/contrastive_train_pairs.csv')

    val_df = make_file_csv(val_pairs)
    val_df.to_csv('data/paired_file_paths/contrastive_val_pairs.csv')

    test_df = make_file_csv(test_pairs)
    test_df.to_csv('data/paired_file_paths/contrastive_test_pairs.csv')

            

# create_file_csv()


# create_and_save_dataset()
# load_paired_file_paths()

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