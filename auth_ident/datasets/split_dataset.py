import pandas as pd
import tensorflow as tf
import os

from auth_ident.preprocessing import load_data
from auth_ident.generators import PairGen
import auth_ident
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from bpe import Encoder
import pickle


class SplitDataset:
    def __init__(self,
                 max_code_length,
                 batch_size,
                 encoding_type='bpe',
                 flip_labels=False,
                 language='python'):
        
        self.max_code_length = max_code_length
        self.batch_size = batch_size
        self.encoding_type = encoding_type
        self.flip_labels = flip_labels

        print("\nIn INIT\n", flush=True)
        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        self.start = "<start>"
        self.end = "<end>"
        chars_to_encode = [self.start, self.end] + list(chars_to_encode)
        self.len_encoding = len(chars_to_encode)
        chars_index = [i for i in range(self.len_encoding)]

        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode,
                                                       chars_index,
                                                       key_dtype=tf.string,
                                                       value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map,
                                                     num_oov_buckets=1)

        pickleFile = open(os.path.join('data/loaded/encoders', language + "_encoder.pkl"), 'rb')
        self.encoder = pickle.load(pickleFile)


    def encode_to_one_hot(self, code_to_embed):
        reshaped = tf.concat(
            [[self.start],
             tf.strings.unicode_split(code_to_embed, 'UTF-8'), [self.end]],
            axis=0)
        encoding = self.table.lookup(reshaped)
        encoding = tf.reshape(
            tf.squeeze(tf.one_hot(encoding, self.len_encoding)),
            (-1, self.len_encoding))

        code_length = tf.shape(encoding)[0]
        padding = [[0, self.max_code_length + 2 - code_length], [0, 0]]
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=1)

        return encoding

    def bpe_encode(self, code_to_encode):
        
        lines = [line.splt('\n') for line in code_to_encode]
        self.encoder.transform(lines)

    def flip_labels(self, files, label):
        if label == 0:
            label = 1
        else:
            label = 1
        return files, label

    def create_dataset(self, language, split):

        def bpe_encode_both(files, labels):
            files['input_1'] = self.bpe_encode(files['input_1'])
            files['input_2'] = self.bpe_encode(files['input_2'])

            return files, labels

        def encode_one_hot(files, label):
            files["input_1"] = self.encode_to_one_hot(files["input_1"])
            files["input_2"] = self.encode_to_one_hot(files["input_2"])
            return files, label

        def set_shape(files, label):
            files["input_1"].set_shape(
                (self.max_code_length + 2, self.len_encoding))
            files["input_2"].set_shape(
                (self.max_code_length + 2, self.len_encoding))
            label = label
            return files, label

        def set_batch_shape(files, label):
             
            files["input_1"].set_shape(
                (None, self.max_code_length + 2, self.len_encoding))
            files["input_2"].set_shape(
                (None, self.max_code_length + 2, self.len_encoding))
            label = label
            return files, label

        if split == 'train':
            num_samples = auth_ident.TRAIN_LEN
        elif split == 'val':
            num_samples = auth_ident.VAL_LEN
        elif split == 'test':
            num_samples = auth_ident.TEST_LEN
        else:
            print(
                "ERROR: Invalid split type in split_dataset.create_dataset: " +
                split)

        f = "data/loaded/" + language + "_" + split + ".h5"
        df = pd.read_hdf(f)
        pg = PairGen(df,
                     crop_length=self.max_code_length,
                     samples_per_epoch=num_samples)

        print("Generating Data...", flush=True)
        dataset = tf.data.Dataset.from_generator(
            pg.gen, 
            ({
                "input_1": tf.string,
                "input_2": tf.string
            }, tf.bool),
            output_shapes=(
                {
                    "input_1":
                    tf.TensorShape([]),
                    "input_2":
                    tf.TensorShape([])
                }, 
                tf.TensorShape([]))
        )

        print("Data Generated.", flush=True)

        dataset = dataset.repeat()

        if self.encoding == 'bpe':
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(self.bpe_encode, tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(set_batch_shape, tf.data.experimental.AUTOTUNE)
        elif self.encoding == 'one_hot':
            dataset = dataset.map(encode_one_hot)
            dataset = dataset.map(set_shape, tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(self.batch_size)

        if self.flip_labels:
            print(
                "ERROR: Flip Labels not supported: split_dataset.create_dataset"
            )
            exit(1)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

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
