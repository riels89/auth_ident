import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import itertools
import math

chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
chars_to_encode = list(chars_to_encode)
len_encoding = len(chars_to_encode)
chars_index = [i for i in range(len_encoding)]

char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode, chars_index, key_dtype=tf.string, value_dtype=tf.int64)
table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)

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
    for root, _, files in os.walk("gcj"):
        for file in files:
            filepaths.append(os.path.join(root, file))
            usernames.append(root.split("\\")[2])

    filepaths = pd.DataFrame({"username": usernames, "filepath": filepaths})
    filepaths.to_csv("gcj.csv")


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

def pair_authors_slow(total_samples, match_rate=0.2, train_split=0.8):

    files = pd.read_csv("gcj.csv")

    def generate_pairs(percent_total, authors):
        pairs = []
        labels = []
        for i in range(int(total_samples * percent_total)):
            print("Sample: " + str(i))
            author1 = authors[random.randint(0, len(authors) - 1)]
            author1_files = files.loc[files['username'] == author1]['filepath'].values
            file1 = author1_files[random.randint(0, len(author1_files) - 1)]

            if random.random() > match_rate or len(author1_files) is 1:
                author2 = authors[random.randint(0, len(authors) - 1)]
                while(author1 == author2):
                    author2 = authors[random.randint(0, len(authors) - 1)]
                author2_files = files.loc[files['username'] == author2]['filepath'].values
                file2 = author2_files[random.randint(0, len(author2_files) - 1)]
                pairs.append([file1, file2])
                labels.append(0)
            else:
                author2 = author1
                author2_files = files.loc[files['username'] == author2]['filepath'].values
                file2 = author2_files[random.randint(0, len(author2_files) - 1)]
                while(file2 == file1):
                    file2 = author1_files[random.randint(0, len(author2_files) - 1)]
                pairs.append([file1, file2])
                labels.append(1)

        return pairs, labels

    authors = list(set(files['username']))
    train_authors = authors[:int(train_split * len(authors))]
    val_authors = authors[int(train_split * len(authors)):]

    train_pairs, train_labels = generate_pairs(train_split, train_authors)
    val_pairs, val_labels = generate_pairs(1 - train_split, val_authors)

    return train_pairs, train_labels, val_pairs, val_labels

def pair_authors_fast(total_samples, match_rate=0.2, train_split=0.8):
    """Generates pairs of authors. Author pairs will be pairs of the same author,
    based on the match rate percentage. The rest of the time Author pairs will be
    of different authors."""
    files = pd.read_csv("gcj.csv", keep_default_na=False)
    authors = list(set(files['username']))
    auth_to_idx = {authors[i]: i for i in range(len(authors))}
    idx_to_auth = {v: k for k, v in auth_to_idx.items()}

    author_file_count = files['username'].value_counts()
    authors = [auth_to_idx[author] for author in authors]

    files_by_auth = files.groupby(['username']).indices # dict
    files_by_auth = np.array([np.array(files_by_auth[idx_to_auth[auth_idx]]) for auth_idx in authors])

    non_matched_length = int(total_samples * (1 - match_rate))
    matched_length = int(total_samples * match_rate)

    def match_authors():
        """Gets number of times each author will match with themselvs"""
        authors_to_match = []
        samples_left = matched_length
        index = 1
        while(samples_left > 0):
            # authors hat refers to authors with a file count > index
            authors_h_i_name = author_file_count[author_file_count > index].index
            authors_h_i = [auth_to_idx[author] for author in authors_h_i_name]

            if samples_left > len(authors_h_i):
                authors_to_match.append(authors_h_i)
            else:
                authors_to_match.append(np.random.choice(authors_h_i, samples_left, replace=False))

            samples_left -= len(authors_h_i)
            index += 1
        return authors_to_match

    def get_all_file_combinations():
        # authors hat refers to authors with a file count > 1
        authors_h_name = author_file_count[author_file_count > 1].index
        authors_h = np.sort([auth_to_idx[author] for author in authors_h_name])

        print("num authors: " + str(len(authors)))
        print("num authors file count > 1: " + str(len(authors_h)))

        # Grouping the files by author
        file_h = files.groupby(['username']).indices # dict
        file_h = np.array([file_h[idx_to_auth[auth_idx]] for auth_idx in authors_h])

        # Getting all possible combinations
        file_h_c = np.array([np.array(list(itertools.combinations(file_set, 2)), dtype=(int, int))
                            for file_set in file_h])
        return file_h_c

    def select_matched_files(authors_to_match, file_h_c):
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

    # Get pairs of the same author
    authors_to_match = match_authors()
    file_h_c = get_all_file_combinations()
    matched_combinations = select_matched_files(authors_to_match, file_h_c)

    print(matched_combinations.shape)

    def select_authors():
        """Select a random author, then select random author indices from num authors - 1 because
         we don't want to count the same author twice. Then for each second author index
         that is >= the first author, add one. This ensures we select from the correct
         set of possible values and avoids selecting the same author.
         Ex: The first authors to choose from is:  [1, 2, 3, 4, 5]
         The second authors to choose from is: [1, 2, 3, 4]
         If the first chosen author is 2
         2 >= 2 so 2 => 3. 3 => 4.  4 => 5
         Then the second author actually chooses from: [1, 3, 4, 5]"""
        first_author = np.squeeze(np.random.choice(authors, [non_matched_length, 1]))
        second_author = np.squeeze(np.random.choice(authors[:len(authors) - 1], [non_matched_length, 1]))
        bool_mask = second_author >= first_author
        second_author += bool_mask
        return first_author, second_author

    def choose_files(authors):
        """Maps a set of authors to a set of randomly chosen files"""
        files_len = np.array([files_by_auth[author].shape[0] for author in authors])
        chosen_file = (np.random.rand(*authors.shape) * np.squeeze(files_len)).astype(int)
        files = files_by_auth[authors]
        files = np.array([files[i][chosen_file[i]] for i in range(non_matched_length)])
        return files

    # Get pairs of different authros
    first_author, second_author = select_authors()
    first_files = choose_files(first_author)
    second_files = choose_files(second_author)
    non_matched_combinations = np.column_stack([first_files, second_files])

    print(non_matched_combinations.shape)

    data = np.concatenate([matched_combinations, non_matched_combinations])
    labels = np.concatenate([np.ones_like(matched_combinations), np.zeros_like(non_matched_combinations)])

    shuffle_mask = np.random.shuffle(np.arange(data.shape[0]))

    data = data[shuffle_mask]
    labels = labels[shuffle_mask]

    train_pairs = data[total_samples * train_split]
    train_labels = data[total_samples * train_split]
    val_pairs = data[total_samples * (1 - train_split)]
    val_labels = data[total_samples * (1 - train_split)]

    return X_train, y_train, X_val, y_val

def encode_to_characters(code_to_embed):
    # start = tf.timestamp(name=None)

    reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
    encoding = table.lookup(reshaped)
    encoding = tf.squeeze(tf.one_hot(encoding, len_encoding))

    code_length = tf.shape(encoding)[0]
    padding = [[0, max_code_length - code_length], [0, 0]]
    encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=0)

    # end = tf.timestamp(name=None)
    # tf.print("Embedding time: ", [end - start])

    return encoding


def get_dataset(batch_size, seed=13):

    def encode(code_to_embed, username):
        return encode_to_characters(encoding), username

    def get_file(file):
        # start = tf.timestamp(name=None)

        parts = tf.strings.split(file, '/')
        output = (tf.io.read_file(file), parts[2])

        # end = tf.timestamp(name=None)
        # tf.print("Get file time: ", [end - start])

        return output

    files = pd.read_csv("gcj.csv")

    dataset = tf.data.Dataset.from_tensor_slices(files['filepath'])
    dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda code, label: tf.strings.length(code) <= max_code_length)
    dataset = dataset.map(encode, tf.data.experimental.AUTOTUNE)

    # dataset = dataset.map(tf_file_stats)

    dataset = dataset.shuffle(1024, seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def get_dataset_pairs(batch_size, seed=13):

    def encode(file1, file2, label):
        return encode_to_characters(file1), encode_to_characters(file2), label

    def get_file(files, label):
        # start = tf.timestamp(name=None)

        output = (tf.io.read_file(files[0]), tf.io.read_file(files[1]), label)

        # end = tf.timestamp(name=None)
        # tf.print("Get file time: ", [end - start])

        return output

    train_pairs, train_labels, val_pairs, val_labels = pair_authors_fast(1e6)

    dataset = tf.data.Dataset.from_tensor_slices((train_pairs, train_labels))
    dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda file1, file2, label: tf.strings.length(file1) <= max_code_length and tf.strings.length(file2) <= max_code_length)
    dataset = dataset.map(encode, tf.data.experimental.AUTOTUNE)

    # dataset = dataset.map(tf_file_stats)

    dataset = dataset.shuffle(1024, seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


# dataset = get_dataset_pairs(128)
#
# for batch in dataset.take(1):
#     print(batch)
pair_authors_fast(1e7)
# print("Embedding time: " + str(sum(embedding_times) / tf.shape(embedding_times)[0]))
# print("Get file time: " + str(sum(get_file_times) / tf.shape(get_file_times)[0]))
# print(embedding_times)
# make_csv()