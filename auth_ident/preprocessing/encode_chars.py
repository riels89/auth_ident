from os.path import join
import os
import argparse
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
tqdm.pandas()

chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
start = "<start>"
end = "<end>"
chars_to_encode = [end, start] + list(chars_to_encode)

len_encoding = len(chars_to_encode) + 1

chars_index = [i for i in range(len(chars_to_encode))]

char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode,
                                               chars_index,
                                               key_dtype=tf.string,
                                               value_dtype=tf.int64)
table = tf.lookup.StaticVocabularyTable(char_map, num_oov_buckets=1)


def encode(sentence):
    reshaped = tf.concat(
        [[start], tf.strings.unicode_split(sentence, 'UTF-8'), [end]], axis=0)
    encoding = tf.cast(table.lookup(reshaped), dtype=tf.int8)

    return encoding


def encode_data(data_file):

    split_path = data_file.split('/')
    loaded_dir = '/'.join(split_path[:-2])
    file_name = "char_encoding"

    os.makedirs(join(loaded_dir, 'char_encoded_data'), exist_ok=True)

    train_data = pd.read_hdf(data_file + "_train.h5")

    train_data['file_content'] = train_data['file_content'].progress_apply(
        encode, convert_dtype=False)
    train_output_file = join(loaded_dir, "char_encoded_data",
                             f"{file_name}_train.h5")
    print(f"Output train file: {train_output_file}")
    train_data.to_hdf(train_output_file, key='data', mode='w')

    val_data = pd.read_hdf(data_file + "_val.h5")
    val_data['file_content'] = val_data['file_content'].progress_apply(
        encode, convert_dtype=False)

    val_output_file = join(loaded_dir, "char_encoded_data",
                           f"{file_name}_val.h5")
    print(f"Output val file: {val_output_file}")
    val_data.to_hdf(val_output_file, key='data', mode='w')

    test_data = pd.read_hdf(data_file + "_test.h5")
    test_data['file_content'] = test_data['file_content'].progress_apply(
        encode, convert_dtype=False)

    test_output_file = join(loaded_dir, "char_encoded_data",
                            f"{file_name}_test.h5")
    print(f"Output test file: {test_output_file}")
    test_data.to_hdf(test_output_file, key='data', mode='w')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_file', help='spm model file')

    args = parser.parse_args()

    encode_data(args.data_file)


if __name__ == "__main__":
    main()
