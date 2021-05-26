import pandas as pd
import tensorflow as tf
from os.path import join
import os

from auth_ident.generators import SimCLRGen
import sentencepiece as spm
import auth_ident
from auth_ident import CPP_JAVA_INDEX_BUFFER


class SimCLRDataset:
    def __init__(self,
                 max_code_length,
                 batch_size,
                 data_file=None,
                 encoding_type='char',
                 spm_model_file=None):
        print("\nIn INIT\n", flush=True)

        self.max_code_length = max_code_length
        self.batch_size = batch_size
        self.data_file = data_file
        self.encoding_type = encoding_type

        chars_to_encode = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM\n\r\t " + r"1234567890-=!@#$%^&*()_+[]{}|;':\",./<>?"
        self.start = "<start>"
        self.end = "<end>"
        chars_to_encode = [self.start, self.end] + list(chars_to_encode)
        
        if self.encoding_type == "spm":
            sp = spm.SentencePieceProcessor(model_file=spm_model_file)
            self.len_encoding = sp.vocab_size()
        elif self.encoding_type == "tokens":
            
            if data_file.str.contains("cpp"):
                self.len_encoding = CPP_JAVA_INDEX_BUFFER
            elif data_file.str.contains("java"):
                self.len_encoding = CPP_JAVA_INDEX_BUFFER
            else:
                assert False, "No python length encoding known"

            # Assume format data/.../{language}_{type}_encoded.h5
            top_ids_file = "_".join(data_file.split("_")[:-2]) + "_top_identifiers"
            with open(top_ids_file) as f:
                num_reserved_identifiers = sum([1 for line in f])

            self.len_encoding += num_reserved_identifiers
        else:
            self.len_encoding = len(chars_to_encode) + 1

        chars_index = [i for i in range(len(chars_to_encode))]

        char_map = tf.lookup.KeyValueTensorInitializer(chars_to_encode,
                                                       chars_index,
                                                       key_dtype=tf.string,
                                                       value_dtype=tf.int64)
        self.table = tf.lookup.StaticVocabularyTable(char_map,
                                                     num_oov_buckets=1)

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
        encoding = tf.pad(encoding, padding, 'CONSTANT', constant_values=0)

        return encoding

    def create_dataset(self):

        def encode_one_hot(files, label):
            files["input_1"] = self.encode_to_one_hot(files["input_1"])
            files["input_2"] = self.encode_to_one_hot(files["input_2"])
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

        f = join("data/organized_hdfs/", self.data_file)
        print(os.path.exists(f))
        print(f)
        df = pd.read_hdf(f)
        pg = SimCLRGen(df,
                       crop_length=self.max_code_length,
                       batch_size=self.batch_size,
                       samples_per_epoch=num_samples)

        print("Generating Data...", flush=True)

        if self.encoding_type == 'spm':
            shape = [self.max_code_length]
        elif self.encoding_type == 'char':
            shape = [self.max_code_length]

        dataset = tf.data.Dataset.from_generator(pg.gen, ({
            "input_1": tf.int32,
            "input_2": tf.int32
        }, tf.bool),
            output_shapes=({
                "input_1":
                tf.TensorShape(shape),
                "input_2":
                tf.TensorShape(shape)
            }, tf.TensorShape([])))

        print("Data Generated.", flush=True)

        dataset = dataset.repeat()

        print(f"batch_size {self.batch_size}")
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    sds = SimCLRDataset(20, 4)
    train_dataset, val_dataset, test_dataset = sds.get_dataset()
    for i in val_dataset.take(1):
        print("I", i)
        print("INPUT SHAPE", i['input_1'].shape)
