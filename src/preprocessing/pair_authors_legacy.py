import pandas as pd
import numpy as np
import itertools


def pair_authors_fast(total_samples, match_rate=0.2, train_samples=1000000, val_test_samples=20000):
    """Generates pairs of authors. Author pairs will be pairs of the same author,
    based on the match rate percentage. The rest of the time Author pairs will be
    of different authors."""
    files = pd.read_csv("refrences/gcj.csv", keep_default_na=False)
    authors = list(set(files['username']))
    auth_to_idx = {authors[i]: i for i in range(len(authors))}
    idx_to_auth = {v: k for k, v in auth_to_idx.items()}

    author_file_count = files['username'].value_counts()
    authors = [auth_to_idx[author] for author in authors]

    files_by_auth = files.groupby(['username']).indices # dict
    files_by_auth = np.array([np.array(files_by_auth[idx_to_auth[auth_idx]]) for auth_idx in authors])

    non_matched_length = int(total_samples * (1 - match_rate))
    matched_length = int(total_samples * match_rate)

    def match_authors():
        """Gets number of times each author will match with themselvs"""
        authors_to_match = []
        samples_left = matched_length
        index = 1
        while(samples_left > 0):
            # authors hat refers to authors with a file count > index
            authors_h_i_name = author_file_count[author_file_count > index].index
            authors_h_i = [auth_to_idx[author] for author in authors_h_i_name]

            if samples_left > len(authors_h_i):
                authors_to_match.append(authors_h_i)
            else:
                authors_to_match.append(np.random.choice(authors_h_i, samples_left, replace=False))

            samples_left -= len(authors_h_i)
            index += 1
        return authors_to_match

    def get_all_file_combinations():
        # authors hat refers to authors with a file count > 1
        authors_h_name = author_file_count[author_file_count > 1].index
        authors_h = np.sort([auth_to_idx[author] for author in authors_h_name])

        print("num authors: " + str(len(authors)))
        print("num authors file count > 1: " + str(len(authors_h)))

        # Grouping the files by author
        file_h = files.groupby(['username']).indices # dict
        file_h = np.array([file_h[idx_to_auth[auth_idx]] for auth_idx in authors_h])

        # Getting all possible combinations
        file_h_c = np.array([np.array(list(itertools.combinations(file_set, 2)), dtype=(int, int))
                            for file_set in file_h])
        return file_h_c

    def select_matched_files(authors_to_match, file_h_c):
        """Maps the given authors to a random combination of files, without duplicates"""
        # Bin duplicate authors, the bin index is replaced by their count.
        # authors_to_match:=[1, 2, 2, 2, 3, 3] => [0, 1, 1, 1, 2, 2] => [1, 3, 2] = auth_bin_cnt
        authors_to_match = np.sort(np.concatenate(authors_to_match))
        authors_to_match_set, set_inverse = np.unique(authors_to_match, return_inverse=True)
        auth_bin_cnt = np.bincount(set_inverse)

        combinations_idx = []
        for author in range(len(auth_bin_cnt)):
            combinations_idx.append(np.random.choice(len(file_h_c[author]), auth_bin_cnt[author], replace=False))

        # Convert to numpy array
        matched_combinations = np.array([np.array(file_h_c[author][idx])
                                        for author in range(len(combinations_idx))
                                        for idx in combinations_idx[author]])
        return matched_combinations

    # Get pairs of the same author
    authors_to_match = match_authors()
    file_h_c = get_all_file_combinations()
    matched_combinations = select_matched_files(authors_to_match, file_h_c)

    print("Same author pairing shape: " + str(matched_combinations.shape))

    def select_authors():
        """Select a random author, then select random author indices from num authors - 1 because
         we don't want to count the same author twice. Then for each second author index
         that is >= the first author, add one. This ensures we select from the correct
         set of possible values and avoids selecting the same author.
         Ex: The first authors to choose from is:  [1, 2, 3, 4, 5]
         The second authors to choose from is: [1, 2, 3, 4]
         If the first chosen author is 2
         2 >= 2 so 2 => 3. 3 => 4.  4 => 5
         Then the second author actually chooses from: [1, 3, 4, 5]"""
        first_author = np.squeeze(np.random.choice(authors, [non_matched_length, 1]))
        second_author = np.squeeze(np.random.choice(authors[:len(authors) - 1], [non_matched_length, 1]))
        bool_mask = second_author >= first_author
        second_author += bool_mask
        return first_author, second_author

    def choose_files(authors):
        """Maps a set of authors to a set of randomly chosen files"""
        files_len = np.array([files_by_auth[author].shape[0] for author in authors])
        chosen_file = (np.random.rand(*authors.shape) * np.squeeze(files_len)).astype(int)
        files = files_by_auth[authors]
        files = np.array([files[i][chosen_file[i]] for i in range(non_matched_length)])
        return files

    # Get pairs of different authros
    first_author, second_author = select_authors()
    first_files = choose_files(first_author)
    second_files = choose_files(second_author)
    non_matched_combinations = np.column_stack([first_files, second_files])

    print("Different author pairings shape: " + str(non_matched_combinations.shape))

    data = np.concatenate([matched_combinations, non_matched_combinations])
    labels = np.concatenate([np.ones(matched_combinations.shape[0]), np.zeros(non_matched_combinations.shape[0])])

    shuffle_mask = np.random.permutation(data.shape[0])
    data = data[shuffle_mask]
    data = files['filepath'].take(data.flatten()).values.reshape(-1, 2)
    labels = np.squeeze(labels[shuffle_mask])

    train_pairs = data[:train_samples]
    train_labels = labels[:train_samples]
    val_pairs = data[train_samples:train_samples + val_test_samples]
    val_labels = labels[train_samples:train_samples + val_test_samples]
    test_pairs = data[train_samples:train_samples + 2 * val_test_samples]
    test_labels = labels[train_samples:train_samples + 2 * val_test_samples]

    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels
