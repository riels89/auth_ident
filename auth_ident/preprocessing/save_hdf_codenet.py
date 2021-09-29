"""
Command line script for processing the directory hierarchy
containing IBM CodeNet submissions to create h5 files that can be read
by Pandas.  The columns in the resulting pandas dataframes will be

"problem_id"
"user_id"
"language"
"filepath"
"file_content" - The full file contents. (Optional)

"""

import argparse
import os
import re
import numpy as np
import pandas as pd


def get_dups(codenet_path, language):
    """Return a set containing all duplicate (or near duplicate)
       submission ids. The first entry from each cluster will not be
       included in this set.
    """
    lang_path = os.path.join(codenet_path, 'derived', 'duplicates', language)
    file_name = 'Project_CodeNet-' + language + ".clusters"
    dups = set()
    regex_str = '.*/{}/(.*)\..*:.*'.format(re.escape(language))
    regex = re.compile(regex_str)

    with open(os.path.join(lang_path, file_name), 'r') as cluster_file:

        cur_line = "dummy"
        while cur_line:
            cur_line = cluster_file.readline()  # first line from a cluster
            if cur_line:
                cur_line = cluster_file.readline()  # first dup (or empty line)
                while cur_line and cur_line != "\n":
                    m = regex.match(cur_line)
                    dups.add(m.group(1))  # submission id
                    cur_line = cluster_file.readline()
    return dups


def make_hdf(codenet_root, new_hdf, keep_repeats, languages, val_test_split,
             min_file_size, dataset, include_contents):
    """
    Create .h5 file(s) from Google code jam submissions.

    :param codenet_root (srting): Root of the code jam file hierarchy.
    :param new_hdf (string): Filename to save (.h5 will be appended)
    :param keep_repeats (boolean): Keep multiple problem submissions.
    :param languages (list of strings): Keep files with these languages.
    :param val_test_split (float): Fraction of files to set aside for
    each of validation and testing.
    :param min_file_size (int): Minimum file size to keep (in bytes).
    :param dataset (string): Restrict to single dataset AIZU or AtCoder.
    :param include_contents (boolean): Include full file contents.

    """
    submissions = []
    codenet_root = os.path.normpath(codenet_root)
    metadata_root = os.path.join(codenet_root, 'metadata')
    data_root = os.path.join(codenet_root, 'data')

    if not keep_repeats:
        dups = set()
        for language in languages:
            dups.update(get_dups(codenet_root, language))

    problems_df = pd.read_csv(os.path.join(metadata_root,
                                           'problem_list.csv'))
    if dataset:
        prob_ids = list(
            problems_df.loc[problems_df['dataset'] == dataset]['id'])
    else:
        prob_ids = list(problems_df['id'])

    prob_csvs = [prob_id + '.csv' for prob_id in prob_ids]

    total_checked = 0
    size_rejected = 0
    dup_rejected = 0
    for prob_num, csv_file in enumerate(prob_csvs):

        print("Processing {} {}/{} ({} files)".format(csv_file,
                                                      prob_num,
                                                      len(prob_csvs),
                                                      len(submissions)))
        metadata_df = pd.read_csv(os.path.join(metadata_root, csv_file))

        for language in languages:
            language_df = metadata_df.loc[metadata_df['language'] == language]
            for _, frame in language_df.iterrows():
                total_checked += 1
                if frame['code_size'] <= min_file_size:
                    size_rejected += 1
                    continue
                if not keep_repeats and frame['submission_id'] in dups:
                    dup_rejected += 1
                    continue

                file_name = (frame['submission_id'] + "."
                             + frame['filename_ext'])
                local_path = os.path.join(frame['problem_id'],
                                          language, file_name)
                full_path = os.path.join(data_root, local_path)

                if include_contents:
                    with open(full_path, 'rb') as content_file:
                        contents = content_file.read().decode("utf-8")
                        submission = (frame['problem_id'],
				      int(frame['user_id'][1:]),
                                      language,
                                      local_path,
                                      contents)
                else:
                    submission = (frame['problem_id'],
                                  int(frame['user_id'][1:]),
                                  language,
                                  local_path)

                submissions.append(submission)


    if include_contents:
        frame = pd.DataFrame({"problem_id": [v[0] for v in submissions],
                              "username": [v[1] for v in submissions],
                              "language": [v[2] for v in submissions],
                              "filepath": [v[3] for v in submissions],
                              "file_content": [v[4] for v in submissions]})
    else:
        frame = pd.DataFrame({"problem_id": [v[0] for v in submissions],
                              "username": [v[1] for v in submissions],
                              "language": [v[2] for v in submissions],
                              "filepath": [v[3] for v in submissions]})

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

    return total_checked, size_rejected, dup_rejected


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    """ Trick to allow both defaults and nice formatting in the help. """
    pass


def main():
    description = '''Process a directory hierarchy containing IBM CodeNet 
submissions to create h5 files that can be read by Pandas.  
The columns in the resulting pandas dataframes will be

"problem_id"
"username"
"language"
"file_content" - The full file contents.
'''

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=description)

    parser.add_argument('--src-dir',
                        default='/home/spragunr/codenet/full/Project_CodeNet/',
                        help='Root directory of codenet dataset.')
    parser.add_argument('--keep-repeats', default=False, action='store_true',
                        help='Keep duplicate/near duplicate files.')
    parser.add_argument('--include-contents', default=False, action='store_true',
                        help='Include full file contents.')
    parser.add_argument('--out', required=True, help='Destination file. (.h5 '
                                                     'will be appended)')
    parser.add_argument('--languages', default=['Python'], nargs='*',
                        help="List of file languages to keep."
                             " (C, C++, Java, Python, etc.)")
    parser.add_argument('--val-test-split', default=0, type=float,
                        help='Fraction of files to set aside for each of '
                             'validation and testing.  If greater than zero, '
                             'three files will be created with _val _test '
                             'and _train appended to the names.')
    parser.add_argument('--min-file-size', default=100, type=int,
                        help='Minimum file size (in bytes) to include.')
    parser.add_argument('--dataset',
                        help='Restrict to just one competition dataset. (AtCoder or AIZU)')

    args = parser.parse_args()
    total, size_rejected, dup_rejected = make_hdf(args.src_dir, args.out,
                                                  args.keep_repeats,
                                                  args.languages,
                                                  args.val_test_split,
                                                  args.min_file_size,
                                                  args.dataset,
                                                  args.include_contents)

    # Print info about the dataset
    with open(args.out + ".info", 'w') as f:
        print("Keep repeats: ", args.keep_repeats, file=f)
        print("Languages: ", args.languages, file=f)
        print("Fraction for test/val:", args.val_test_split, file=f)
        print("Min size kept:", args.min_file_size, file=f)
        print("Files searched: ", total, file=f)
        print("Rejected for size: ", size_rejected, file=f)
        print("Duplicates rejected: ", dup_rejected, file=f)
        print("Dataset restriction: ", args.dataset, file=f)


if __name__ == "__main__":
    main()
