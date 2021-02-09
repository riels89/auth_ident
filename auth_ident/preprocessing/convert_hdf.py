from os.path import join
import re
import os
import argparse
import pandas as pd
import sentencepiece as sp
import itertools
from tqdm.auto import tqdm
tqdm.pandas()


def encode_data(model_file, data_file, alpha, length, by_line):

    spm = sp.SentencePieceProcessor(model_file)

    def encode(program):

        null_char_counter = 0
        still_null_char_counter = 0

        if '\0' in program:
            null_char_counter += 1
            print(f'Null char {null_char_counter}')

        null_char_regex = re.compile('\0')
        null_char_repl = null_char_regex.sub("", program)
        if '\0' in null_char_repl:
            still_null_char_counter += 1
            print(f'Still null char {null_char_counter}')

        tab_regex = re.compile(r'\t')
        tab_repl = tab_regex.sub("[TAB]", null_char_repl)

        if not by_line:
            _newline_regex = re.compile(r"\n")
            eol_processed = _newline_regex.sub(r"[EOL]", tab_repl)
        else:
            eol_processed = [split + '[EOL]\n' for split in tab_repl.split("\n")]

        encoded = spm.encode(eol_processed,
                             alpha=alpha,
                             nbest_size=length,
                             out_type=int,
                             enable_sampling=True)

        if by_line:
            encoded = list(itertools.chain.from_iterable(encoded))

        return encoded

    split_path = model_file.split('/')
    loaded_dir = '/'.join(split_path[:-2])
    file_name = split_path[-1].split('.')[0]

    os.makedirs(join(loaded_dir, 'encoded_data'), exist_ok=True)

    train_data = pd.read_hdf(data_file + "_train.h5")

    train_data['file_content'] = train_data['file_content'].progress_apply(
        encode, convert_dtype=False)

    train_output_file = join(loaded_dir, "encoded_data",
                             f"{file_name}_a{alpha}_l{length}_train.h5")
    print(f"Output train file: {train_output_file}")
    train_data.to_hdf(train_output_file, key='data', mode='w')

    val_data = pd.read_hdf(data_file + "_val.h5")
    val_data['file_content'] = val_data['file_content'].progress_apply(
        encode, convert_dtype=False)

    val_output_file = join(loaded_dir, "encoded_data",
                           f"{file_name}_a{alpha}_l{length}_val.h5")
    print(f"Output val file: {val_output_file}")
    val_data.to_hdf(val_output_file, key='data', mode='w')

    test_data = pd.read_hdf(data_file + "_test.h5")
    test_data['file_content'] = test_data['file_content'].progress_apply(
        encode, convert_dtype=False)

    test_output_file = join(loaded_dir, "encoded_data",
                            f"{file_name}_a{alpha}_l{length}_test.h5")
    print(f"Output test file: {test_output_file}")
    test_data.to_hdf(test_output_file, key='data', mode='w')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_start', help='start of the spm model filepath \
        that you want (can give entire path if you only want to use one model)')
    parser.add_argument('-data_start', help='Start of data file to use (can give \
        entire path if you only want to use one data file)')
    parser.add_argument('-alpha',
                        help='The alpha to use for encoding',
                        type=float)
    parser.add_argument('--length',
                        default=-1,
                        help='The length to use for encoding',
                        type=int)

    args = parser.parse_args()

    model_split = args.model_start.split('/')
    model_dir = '/'.join(model_split[:-1])
    model_prefix = model_split[-1]

    data_split = args.data_start.split('/')
    data_dir = "/".join(data_split[:-1])
    data_prefix = data_split[-1]

    for model_file in os.listdir(model_dir):
        if model_file.startswith(model_prefix) and model_file.endswith(".model"):
            by_line = "by_line" in model_file
            for data_file in os.listdir(data_dir):
                if data_file.startswith(data_prefix) and data_file.endswith(".h5"):
                    data_file = '_'.join(data_file.split('_')[:-1])
                    full_model_path = join(model_dir, model_file)
                    full_data_path = join(data_dir, data_file)
                    print(f"Encoding with SPM model : {full_model_path} on Dataset: {full_data_path}")
                    encode_data(full_model_path,
                                full_data_path,
                                alpha=args.alpha,
                                length=args.length,
                                by_line=by_line)


if __name__ == "__main__":
    main()
