"""
Command line script for processing a directory hierarchy containing Google Code
Jam submissions to create a single hdf file that can be read by Pandas.  The
columns in the resulting pandas dataframe will be:

"username" - The user id of the author.
"filepath" - The original relative path for the file.
"file_content" - The full file contents in unicode.


usage: save_hdf.py [-h] [--src-dir SRC_DIR] [--keep-repeats] --out OUT
                   [--extensions [EXTENSIONS [EXTENSIONS ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --src-dir SRC_DIR     Root directory of gcj dataset. (default: data/raw/gcj)
  --keep-repeats        Keep multiple submissions from the same author for the
                        same problem (default: False)
  --out OUT             Destination file. (default: None)
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
import pandas as pd
from bs4 import UnicodeDammit


def make_hdf(gcj_root, new_hdf, keep_repeats, extensions):
    # mapping from "contest_id/username/problem_id/solution_id"
    #        or    "contest_id/username/problem_id" if dropping repeats
    # to a tuple (username, filepath, file_contents)
    submissions = {}
    file_count = 0
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
                    submissions[submission_key] = (local_path.split('/')[1],
                                                   os.path.join(local_path,
                                                                file),
                                                   dammit.unicode_markup)

    frame = pd.DataFrame({"username": [v[0] for v in submissions.values()],
                          "filepath": [v[1] for v in submissions.values()],
                          "file_content": [v[2] for v in
                                           submissions.values()]})

    frame.to_hdf(new_hdf, key='df', mode='w')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    description = '''Process a directory hierarchy containing Google Code Jam submissions to create
a single hdf file that can be read by Pandas.  The columns in the resulting
pandas dataframe will be

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
    parser.add_argument('--out', required=True, help="Destination file.")
    parser.add_argument('--extensions', default=['py'], nargs='*',
                        help="List of file extensions to keep.")

    args = parser.parse_args()
    make_hdf(args.src_dir, args.out, args.keep_repeats, args.extensions)


if __name__ == "__main__":
    main()
