import pandas as pd
import numpy as np
import os
import tensorflow as tf
from bs4 import UnicodeDammit
from auth_ident.preprocessing import PairAuthors
from auth_ident.preprocessing import ContrastiveAuthorComb

# max = 102400
max_chars = 0
counter = 0


def make_csv(gcj_root="data/raw/gcj", new_csv="refrences/gcj.csv"):
    # ->contest_id/
    # --->username/
    # ------>problem_id/
    # -------->solution_id/
    # ---------->extracted/
    # ------------>contestant submitted files

    filepaths = []
    usernames = []

    for root, _, files in os.walk(gcj_root):
        for file in files:
            root_norm = os.path.normpath(root)
            local_path = root_norm[len(gcj_root):]
            filepaths.append(os.path.join(root_norm, file))
            usernames.append(local_path.split('/')[2])

    frame = pd.DataFrame({"username": usernames, "filepath": filepaths})
    frame.to_csv(new_csv, index=False)


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
    pa = PairAuthors("refrences/gcj.csv")
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = pa.generate_pairs()

    np.save('data/paired_file_paths/train_pairs.npy', train_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/train_labels.npy', train_labels, allow_pickle=True)
    np.save('data/paired_file_paths/val_pairs.npy', val_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/val_labels.npy', val_labels, allow_pickle=True)
    np.save('data/paired_file_paths/test_pairs.npy', test_pairs, allow_pickle=True)
    np.save('data/paired_file_paths/test_labels.npy', test_labels, allow_pickle=True)


def create_file_csv():
    pa = ContrastiveAuthorComb("refrences/gcj.csv")
    train_pairs, val_pairs, test_pairs = pa.generate_pairs()
    files = pd.read_csv("refrences/gcj.csv", keep_default_na=False)
    authors = list(set(files['username']))
    auth_to_idx = {authors[i]: i for i in range(len(authors))}

    def make_file_csv(pair_set):
        files = {"file1": [], "file2": [], "author1": [], "author2": []}
        # files = np.empty([pair_set.shape[0], 4], dtype='object, object, i8, i8')

        for i in range(pair_set.shape[0]):
            try:
                if i % 1000:
                    print(i, '/', pair_set.shape[0])
                with open(pair_set[i][0], 'r', errors='surrogateescape') as file:
                    code = file.read()
                    files["file1"].append(code)
                with open(pair_set[i][1], 'r', errors='surrogateescape') as file:
                    code = file.read()
                    files["file2"].append(code)
                files["author1"].append(auth_to_idx[pair_set[i][0].split('/')[4]])
                files["author2"].append(auth_to_idx[pair_set[i][1].split('/')[4]])
            except Exception as e:
                print(e)

        return pd.DataFrame(files)

    train_df = make_file_csv(train_pairs)
    train_df.to_csv('data/paired_file_paths/contrastive_train_pairs.csv')

    val_df = make_file_csv(val_pairs)
    val_df.to_csv('data/paired_file_paths/contrastive_val_pairs.csv')

    test_df = make_file_csv(test_pairs)
    test_df.to_csv('data/paired_file_paths/contrastive_test_pairs.csv')

            
if __name__ == "__main__":
    create_file_csv()

# pa = contrastive_author_comb("refrences/gcj.csv")
# train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = pa.generate_pairs()

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
