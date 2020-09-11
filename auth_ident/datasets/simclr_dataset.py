import pandas as pd
import tensorflow as tf
from os.path import join

from auth_ident.generators import SimCLRGen
import auth_ident


class SimCLRDataset:
    def __init__(self,
                 max_code_length,
                 batch_size,
                 binary_encoding=False,
                 data_file=None):
        print("\nIn INIT\n", flush=True)

        self.data_file = data_file

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

        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode,
                                                       chars_index,
                                                       key_dtype=tf.string,
                                                       value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map,
                                                     num_oov_buckets=1)

        self.max_code_length = max_code_length
        self.batch_size = batch_size

        self.bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)

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

    def encode_to_binary(self, code_to_embed):
        reshaped = tf.strings.unicode_split(code_to_embed, 'UTF-8')
        encoding = tf.cast(self.table.lookup(reshaped) + 1, tf.uint8)
        unpacked = tf.reshape(tf.math.floormod(
            tf.cast(encoding[:, None] // self.bits, tf.int32), 2),
                              shape=(-1, self.binary_encoding_len))

        code_length = tf.shape(unpacked)[0]
        padding = [[0, self.max_code_length - code_length], [0, 0]]
        encoding = tf.pad(unpacked, padding, 'CONSTANT', constant_values=1)

    def create_dataset(self):
        def encode_binary(files, label):
            files["input_1"] = self.encode_to_binary(files["input_1"])
            files["input_2"] = self.encode_to_binary(files["input_2"])
            return files, label

        def encode_one_hot(files, label):
            files["input_1"] = self.encode_to_one_hot(files["input_1"])
            files["input_2"] = self.encode_to_one_hot(files["input_2"])
            return files, label

        def set_shape(files, label):
            files["input_1"].set_shape(
                (self.max_code_length + 2, self.len_encoding))
            files["input_2"].set_shape(
                (self.max_code_length + 2, self.len_encoding))
            return files, label

        if 'train' in self.data_file:
            num_samples = auth_ident.TRAIN_LEN
        elif 'val' in self.data_file:
            num_samples = auth_ident.VAL_LEN
        elif 'test' in self.data_file:
            num_samples = auth_ident.TEST_LEN
        else:
            print(
                r"ERROR: Invalid data_file type (has to contain 'train', 'val', or 'test'):",
                self.data_file)
            exit(1)

        f = join("data/loaded/", self.data_file)
        df = pd.read_hdf(f)
        pg = SimCLRGen(df,
                       crop_length=self.max_code_length,
                       batch_size=self.batch_size,
                       samples_per_epoch=num_samples)

        print("Generating Data...", flush=True)

        dataset = tf.data.Dataset.from_generator(pg.gen, ({
            "input_1": tf.string,
            "input_2": tf.string
        }, tf.bool),
            output_shapes=({
                "input_1":
                tf.TensorShape([]),
                "input_2":
                tf.TensorShape([])
            }, tf.TensorShape([])))

        print("Data Generated.", flush=True)

        dataset = dataset.repeat()

        if self.binary_encoding:
            dataset = dataset.map(encode_binary)
        else:
            dataset = dataset.map(encode_one_hot)

        # dataset = dataset.map(set_shape)

        dataset = dataset.map(set_shape, 120)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    sds = SimCLRDataset(20, 4)
    train_dataset, val_dataset, test_dataset = sds.get_dataset()
    for i in val_dataset.take(1):
        print("I", i)
        print("INPUT SHAPE", i['input_1'].shape)
