import tensorflow as tf
import pandas as pd
import numpy as np
files = pd.read_csv("refrences/gcj.csv", keep_default_na=False)

def get_file(file):
    return tf.io.read_file(file), file

def check_len(file, path):
    return tf.strings.length(file) < 150


dataset = tf.data.Dataset.from_tensor_slices(files['filepath'])
dataset = dataset.shuffle(10000)
dataset = dataset.map(get_file, tf.data.experimental.AUTOTUNE)
dataset = dataset.filter(check_len)
dataset = dataset.map(lambda file, path: path)
dataset = dataset.batch(10000)
dataset = dataset.prefetch(1)

low_char_paths = np.array([])
for batch in dataset:
    low_char_paths = np.concatenate([low_char_paths, batch])

np.save('low_char_paths.npy', low_char_paths, allow_pickle=True)