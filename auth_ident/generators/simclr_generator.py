import pandas as pd
import numpy as np


def just_multi_authors(frame):
    files_by_auth_name = frame.groupby(['username']).indices
    matches = [len(files_by_auth_name[name]) > 1 for name in frame['username']]
    return frame.loc[matches].reset_index(drop=True)


class SimCLRGen:
    """
    Code for creating on-the-fly random file pairings for Sim-CLR training.
    """
    def __init__(self,
                 dataframe,
                 crop_length,
                 batch_size=64,
                 samples_per_epoch=1000):
        self.samples_per_epoch = samples_per_epoch
        self.crop_length = crop_length
        self.batch_size = batch_size
        self.num_batches = samples_per_epoch // self.batch_size

        self.rng = np.random.default_rng(1)

        self.dataframe = just_multi_authors(dataframe)

        self.num_files = len(self.dataframe)

        self.authors = np.array(list(set(self.dataframe['username'])))

        # Mapping from author names to file index
        self.files_by_auth_name = self.dataframe.groupby(['username']).indices

        file_counts = np.array(
            [len(self.files_by_auth_name[name]) for name in self.authors])
        self.author_probs = file_counts / self.num_files
        assert np.isclose(np.sum(self.author_probs), 1.0)

    def gen(self):
        """
        Generate file pairings where each file is equally likely to be
        included in a pairing.

        randomly choose batch_size authors according to distribution
        randomly choose two files by each author

        """
        index = 0  # index into the current batch
        for _ in range(self.samples_per_epoch):
            if index == 0:
                # build the entire batch...
                rand_auth = self.rng.choice(self.authors,
                                            self.batch_size,
                                            replace=False,
                                            p=self.author_probs,
                                            shuffle=False)
                str_type = np.dtype([('crop', np.unicode_, self.crop_length)])
                input_1 = np.empty((self.batch_size, ), dtype=str_type['crop'])
                input_2 = np.empty((self.batch_size, ), dtype=str_type['crop'])

                for i, auth in enumerate(rand_auth):
                    rand_pair = self.rng.choice(self.files_by_auth_name[auth],
                                                2,
                                                replace=False,
                                                shuffle=False)
                    input_1[i] = self.random_crop(rand_pair[0],
                                                  self.crop_length)
                    input_2[i] = self.random_crop(rand_pair[1],
                                                  self.crop_length)

            yield ({'input_1': input_1[index], 'input_2': input_2[index]}, 1)
            index = (index + 1) % self.batch_size

    def random_crop(self, file_indx, crop_length):
        """
        Return a random crop from the file at the provided index. If
        crop_length is longer than the length of the file, then the entire
        file will be returned.
        """

        contents = self.dataframe['file_content'][file_indx]
        if len(contents) > crop_length:
            start = self.rng.integers(0, len(contents) - crop_length + 1)
            contents = contents[start:start + crop_length]
        return contents.ljust(crop_length, '\0')


if __name__ == "__main__":
    import time

    df = pd.read_hdf('/home/spragunr/nobackup/python_test.h5')
    pg = SimCLRGen(df, crop_length=1200, samples_per_epoch=10000)

    start_time = time.perf_counter()
    for _ in range(2):
        for pair in pg.gen():
            print(pair[0])
            # print(".", end="")
            pass

    print("Execution time:", time.perf_counter() - start_time)
