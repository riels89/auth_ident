"""
Command line script for processing a directory hierarchy containing Google Code
Jam submissions to create h5 files that can be read by Pandas.  The columns in
the resultingpandas dataframes will be
"year" - The year of the contest.
"round" - The name of the round, prefixed with an integer for easy sorting.
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
  --min-file-size MIN_FILE_SIZE
                        Minimum file size (in bytes) to include. (default: 100)

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
import json
import numpy as np
import pandas as pd
import pickle
from bs4 import UnicodeDammit


def make_hdf(gcj_root, new_hdf, keep_repeats, extensions, val_test_split,
             min_file_size):
    """
    Create .h5 file(s) from Google code jam submissions.

    :param gcj_root (srting): Root of the code jam file hierarchy.
    :param new_hdf (string): Filename to save (.h5 will be appended)
    :param keep_repeats (boolean): Keep multiple problem submissions.
    :param extensions (list of strings): Keep files with these extensions.
    :param val_test_split (float): Fraction of files to set aside for
    each of validation and testing.
    :param min_file_size (int): Minimum file size to keep (in bytes).

    """
    # mapping from "contest_id/username/problem_id/solution_id"
    #        or    "contest_id/username/problem_id" if dropping repeats
    # to a tuple (username, filepath, file_contents)
    submissions = {}
    file_count = 0
    replaced_files = 0
    loaded_files = 0

    contest_dict = get_competition_data()

    gcj_root = os.path.normpath(gcj_root)
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
            full_path = os.path.join(root, file)
            if (file_extension[1::] in extensions and
                    os.path.getsize(full_path) > min_file_size):

                # pull out just the part of the path from constest_id forward
                local_path = root[len(gcj_root) + 1:]
                split_path = local_path.split('/')

                contest_id = split_path[0]
                problem_id = split_path[2]

                contest_dict, global_problem_index = get_competition_data()

                if keep_repeats:
                    submission_key = '/'.join(
                        split_path[0:4]) + '/' + file
                else:
                    submission_key = '/'.join(
                        split_path[0:3]) + '/' + file

                with open(full_path, 'rb') as content_file:
                    contents = content_file.read()
                    dammit = UnicodeDammit(contents)
                    unicode = dammit.unicode_markup

                    if unicode is not None:
                        # Some contests don't have known problem ids, so we can make them
                        # on the fly
                        problems = contest_dict[contest_id][2]
                        if problem_id in problems.keys(): 
                            problem = problems[problem_id]
                        else:
                            problem = "{:03d} {}".format(global_problem_index, "unknown_name")
                            problems[problem_id] = problem
                            global_problem_index += 1


                        submissions[submission_key] = (contest_dict[contest_id][0], # Year
                                                       contest_dict[contest_id][1], # Round
                                                       problem, # Problem
                                                       local_path.split('/')[1], # Name
                                                       os.path.join(local_path,     # Path
                                                                    file),
                                                       unicode)                     # code
                        loaded_files += 1
                        if dammit.contains_replacement_characters:
                            replaced_files += 1

    frame = pd.DataFrame({"year": [v[0] for v in submissions.values()],
                          "round": [v[1] for v in submissions.values()],
                          "problem": [v[2] for v in submissions.values()],
                          "username": [v[3] for v in submissions.values()],
                          "filepath": [v[4] for v in submissions.values()],
                          "file_content": [v[5] for v in
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


def get_competition_data():
    """ Build a dictionary containing information for each contest id.

    returns: dictionary mapping from contest ids to (year, round)
    tuples.

    """
    script_dir = os.path.dirname(__file__)
    rel_path = "CodeJamMetadata08to17.json"
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path) as f:
        data = json.load(f)

    contest_dict = {}
    global_problem_index = 0

    for competition in data['competitions']:
        year = competition['year']
        for i, round in enumerate(competition['round']):
            problem_dict = {}
            if 'problems' in round.keys():
                for problem in round['problems']:
                    problem_str = '{:03d} {}'.format(global_problem_index, problem["name"])
                    problem_dict[problem["id"]] = problem_str
                    global_problem_index += 1

            round_str = '{:03d} {}'.format(i,round['desc'])
            contest_dict[round['contest']] = (year, round_str, problem_dict)
    return contest_dict, global_problem_index


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
    parser.add_argument('--min-file-size', default=100, type=int,
                        help='Minimum file size (in bytes) to include.')

    args = parser.parse_args()
    total, loaded, replaced = make_hdf(args.src_dir, args.out,
                                       args.keep_repeats, args.extensions,
                                       args.val_test_split, args.min_file_size)

    # Print info about the dataset
    with open(args.out + ".info", 'w') as f:
        print("Keep repeats: ", args.keep_repeats, file=f)
        print("Extensions: ", args.extensions, file=f)
        print("Fraction for test/val:", args.val_test_split, file=f)
        print("Min size kept:", args.min_file_size, file=f)
        print("Files searched: ", total, file=f)
        print("Files loaded: ", loaded, file=f)
        print("Files with replacement chars: ", replaced, file=f)


if __name__ == "__main__":
    main()
