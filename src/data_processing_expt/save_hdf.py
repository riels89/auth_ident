"""
Command line script for processing a directory hierarchy containing Google Code
Jam submissions to create h5 files that can be read by Pandas.  The columns in
the resultingpandas dataframes will be

"username" - The user id of the author.
"filepath" - The original relative path for the file.
"file_content" - The full file contents in unicode.


usage: save_hdf.py [-h] [--src-dir SRC_DIR] [--keep-repeats] --out OUT
                   [--extensions [EXTENSIONS [EXTENSIONS ...]]]
                   [--val-test-split VAL_TEST_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --src-dir SRC_DIR     Root directory of gcj dataset. (default: data/raw/gcj)
  --keep-repeats        Keep multiple submissions from the same author for the
                        same problem (default: False)
  --out OUT             Destination file. (.h5 will be appended) (default:
                        None)
  --val-test-split VAL_TEST_SPLIT
                        Fraction of files to set aside for each of validation
                        and testing. If greater than zero, three files will be
                        created with _val _test and _train appended to the
                        names. (default: 0)

  --extensions [EXTENSIONS [EXTENSIONS ...]]
                        List of file extensions to keep. (default: ['py'])

This script expects that the gcj folders are organized as follows:


    ->contest_id/
    --->username/
    ------>problem_id/
    -------->solution_id/
    ---------->extracted/
    ------------>contestant submitted files

"""

import argparse
import os
import numpy as np
import pandas as pd
from bs4 import UnicodeDammit


def make_hdf(gcj_root, new_hdf, keep_repeats, extensions, val_test_split):
    """
    Create .h5 file(s) from Google code jam submissions.

    :param gcj_root (srting): Root of the code jam file hierarchy.
    :param new_hdf (string): Filename to save (.h5 will be appended)
    :param keep_repeats (boolean): Keep multiple problem submissions.
    :param extensions (list of strings): Keep files with these extensions.
    :param val_test_split (float): Fraction of files to set aside for
    each of validation and testing.

    """
    # mapping from "contest_id/username/problem_id/solution_id"
    #        or    "contest_id/username/problem_id" if dropping repeats
    # to a tuple (username, filepath, file_contents)
    submissions = {}
    file_count = 0
    replaced_files = 0
    loaded_files = 0
    for root, dir_names, files in os.walk(gcj_root):

        # Make sure we walk the subdirectories in order, so later
        # submissions will replace earlier submissions (by default, walk
        # order is arbitrary)
        dir_names[:] = sorted(dir_names)

        for file in files:

            file_count += 1
            if file_count % 1000 == 0:
                print(".", end="", flush=True)

            _, file_extension = os.path.splitext(file)
            if file_extension[1::] in extensions:
                root_norm = os.path.normpath(root)

                # pull out just the part of the path from constest_id forward
                local_path = root_norm[len(gcj_root) + 1:]

                if keep_repeats:
                    submission_key = '/'.join(local_path.split('/')[0:4])
                else:
                    submission_key = '/'.join(local_path.split('/')[0:3])

                full_path = os.path.join(root_norm, file)

                with open(full_path, 'rb') as content_file:
                    contents = content_file.read()
                    dammit = UnicodeDammit(contents)
                    unicode = dammit.unicode_markup

                    if unicode is not None:
                        submissions[submission_key] = (local_path.split('/')[1],
                                                       os.path.join(local_path,
                                                                    file),
                                                       unicode)
                        loaded_files += 1
                        if dammit.contains_replacement_characters:
                            replaced_files += 1


    frame = pd.DataFrame({"username": [v[0] for v in submissions.values()],
                          "filepath": [v[1] for v in submissions.values()],
                          "file_content": [v[2] for v in
                                           submissions.values()]})

    if val_test_split > 0:
        authors = frame['username'].unique()
        np.random.shuffle(authors)

        num_val = int(val_test_split * len(authors))
        num_test = int(val_test_split * len(authors))

        # Determine which authors will be in which set...
        val_authors = np.random.choice(authors, num_val, replace=False)
        authors = np.setdiff1d(authors, val_authors,
                               assume_unique=True)
        test_authors = np.random.choice(authors, num_test, replace=False)

        # All remaining are in the train set.
        train_authors = np.setdiff1d(authors, test_authors,
                                     assume_unique=True)

        # Split into DataFrames according to the selected authors.
        # Unfortunately, this will double memory usage.
        val_matches = [name in val_authors for name in frame['username']]
        test_matches = [name in test_authors for name in frame['username']]
        train_matches = [name in train_authors for name in frame['username']]

        val_frame = frame.loc[val_matches].reset_index(drop=True)
        test_frame = frame.loc[test_matches].reset_index(drop=True)
        train_frame = frame.loc[train_matches].reset_index(drop=True)

        val_frame.to_hdf(new_hdf + "_val.h5", key='df', mode='w')
        test_frame.to_hdf(new_hdf + "_test.h5", key='df', mode='w')
        train_frame.to_hdf(new_hdf + "_train.h5", key='df', mode='w')
    else:
        frame.to_hdf(new_hdf + ".h5", key='df', mode='w')

    return file_count, loaded_files, replaced_files


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """ Trick to allow both defaults and nice formatting in the help. """
    pass


def main():
    description = '''Process a directory hierarchy containing Google Code Jam submissions to create
h5 files that can be read by Pandas.  The columns in the resulting
pandas dataframes will be

"username" - The user id of the author.
"filepath" - The original relative path for the file.
"file_content" - The full file contents in unicode.
    '''

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=description)

    parser.add_argument('--src-dir', default='data/raw/gcj',
                        help='Root directory of gcj dataset.')
    parser.add_argument('--keep-repeats', default=False, action='store_true',
                        help='Keep multiple submissions from the same author '
                             'for the same problem')
    parser.add_argument('--out', required=True, help='Destination file. (.h5 '
                                                     'will be appended)')
    parser.add_argument('--extensions', default=['py'], nargs='*',
                        help="List of file extensions to keep.")
    parser.add_argument('--val-test-split', default=0, type=float,
                        help='Fraction of files to set aside for each of '
                             'validation and testing.  If greater than zero, '
                             'three files will be created with _val _test '
                             'and _train appended to the names.')

    args = parser.parse_args()
    total, loaded, replaced = make_hdf(args.src_dir, args.out, args.keep_repeats, args.extensions,
                                       args.val_test_split)

    # Print info about the dataset
    with open(args.out + ".info", 'w') as f:
        print("Keep repeats: ", args.keep_repeats, file=f)
        print("Extensions: ", args.extenstions, file=f)
        print("Fraction for test/val:", args.val_test_split, file=f)
        print("Files searched: ", total, file=f)
        print("Files loaded: ", loaded, file=f)
        print("Files with replacement chars: ", replaced, file=f)


if __name__ == "__main__":
    main()
