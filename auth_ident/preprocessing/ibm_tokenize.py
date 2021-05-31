import pandas as pd
import subprocess
from tqdm import tqdm
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-src', help='Path of data with language ex: data/gcj_tokenized/c_cpp')
parser.add_argument('--n',
                    default=500,
                    type=int,
                    help="Number of identifiers to reserve")
parser.add_argument(
    '--raw_data_prefix',
    default="data/raw/gcj/",
    help='Prefix to raw data paths from hdf file. Give False if no prefix is needed'
)

parser.add_argument('--l',
                    action="store_true",
                    help='If to load premade tokenization or not')

parser.add_argument('--le',
                    action="store_true",
                    help='If to load premade encoding or not')

args = parser.parse_args()
path = args.src
num_reserved_identifiers = args.n
train_path = path + "_train"
tokenized_train_path = train_path + "_tokenized_csv.csv"


def tokenize(type):
    type_path = path + "_" + type
    tokenized_path = type_path + "_tokenized_csv.csv"

    if not os.path.exists(tokenized_path):
        os.mknod(tokenized_path)
    else:
        os.remove(tokenized_path)
        os.mknod(tokenized_path)

    f = pd.read_hdf(type_path + ".h5")
    if args.raw_data_prefix != "False":
        f["filepath"] = f["filepath"].apply(lambda x: args.raw_data_prefix + x)

    # Save authors for later use in encoding program
    np.savetxt(type_path + "_authors.txt", 
	       f["username"].to_numpy(),
               fmt="%s",
               delimiter="\n")

    command = [
        "./../Project_CodeNet/tools/tokenizer/tokenize", "-lC++", "-mcsv",
        "-n", "-s", "-1", "-a", "-K", f"-o{tokenized_path}"
    ]

    batch_size = 25000
    print(f["filepath"])
    file_paths = f["filepath"].tolist()
    num_batches = int(len(file_paths) / batch_size)
    for batch in tqdm(range(num_batches)):
        command_w_files = command + file_paths[batch * batch_size:batch *
                                               batch_size + batch_size]
        subprocess.call(command_w_files)

    data = pd.read_csv(tokenized_path, encoding="latin1", error_bad_lines=True)
    # Drop unnecessary columns
    data = data[["class", "token"]]
    data = data[data["class"] != "class"]
    data.to_csv(tokenized_path, index=False, header=False)
    print(data)
    del data


if not args.l and not args.le:
    tokenize("train")
    tokenize("val")
    tokenize("test")

    # Find top identifiers
    data = pd.read_csv(tokenized_train_path,
                       encoding="latin1",
                       error_bad_lines=True,
                       header=None,
                       names=["class", "token"])
    tokens = data[data['class'] == "identifier"]["token"].to_numpy(dtype="U20")
    tokens = tokens[:int(tokens.shape[0] / 3)]

    u, counts = np.unique(tokens, return_counts=True)
    # Add 100 length buffer because we will remove individual letters
    sorted_count_args = np.argsort(-counts)[:num_reserved_identifiers + 100]

    # Remove alphabet identifiers because we can deal with that seperately
    alphabet = np.array(
        list(
            "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    top_ids = np.setdiff1d(u[sorted_count_args], alphabet,
                           assume_unique=True)[:num_reserved_identifiers]

    np.savetxt(path + "_top_identifiers.txt",
               top_ids,
               fmt="%s",
               delimiter="\n")


def encode(type):
    if not args.le:
        subprocess.call([
            "auth_ident/preprocessing/encode_tokens",
            f"-n{num_reserved_identifiers}", path + f"_{type }"
        ])

    # Load csv and convert to h5
    f = pd.read_csv(path + f"_{type}_encoded.csv",
                    sep="|",
                    encoding="latin1",
                    converters={
                        "file_content":
                        lambda x: [int(i) for i in x[1:-1].split(",")[:-1]]
                    })
    print(f)
    # Filter out small files
    f = f[f["file_content"].map(len) > 10].reset_index(drop=True)
    f.to_hdf(path + f"_{type}_encoded.h5", key='data', mode='w')


encode("train")
encode("val")
encode("test")
