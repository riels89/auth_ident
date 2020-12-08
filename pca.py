from auth_ident.datasets import ClosedDatset
from auth_ident import GenericExecute, param_mapping
from auth_ident.utils import get_embeddings
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

class AuthorPCA(GenericExecute):

    def execute_one(self, contrastive_params, combination, logger):

        train_data, train_labels, file_indicies = get_embeddings(contrastive_params,
                                                                 ClosedDatset,
                                                                 self.num_authors,
                                                                 self.num_files,
                                                                 "output_embedding",
                                                                 data_file=self.data_file,
                                                                 combination=combination,
                                                                 logger=logger,
                                                                 logdir=self.logdir,
                                                                 return_file_indicies=True)


        f = os.path.join("data/loaded/", self.data_file)
        raw_data = pd.read_hdf(f)

        authors_per_split = int(train_data.shape[0] / float(self.num_files))
        files = np.empty(
            (self.num_authors * self.num_files, train_data.shape[1]))
        labels = np.empty((self.num_authors * self.num_files), dtype=np.int16)
        filepaths = np.empty((self.num_authors * self.num_files), dtype=object)
        for i in range(self.num_files):
            # n_authors_slice = slice(i * self.num_authors, (i + 1) * self.num_authors)
            n_authors_per_split_slice = slice(i * authors_per_split, self.num_authors + i * authors_per_split)
            k_files_slice = slice(i * self.num_authors, (i + 1) * self.num_authors)

            files[k_files_slice] = train_data[n_authors_per_split_slice]
            labels[k_files_slice] = train_labels[n_authors_per_split_slice]
            filepaths[k_files_slice] = raw_data['filepath'][file_indicies[k_files_slice]].values

        labels = np.array([f"Author {i}" for i in labels])
        print(files.shape)
        print(labels.shape)


        pca = PCA(n_components=min(labels.shape[0], train_data.shape[1]))
        components = pca.fit_transform(files)
        print(f"explained variance: {pca.explained_variance_ratio_}")

        data = {
            "PCA Component 0": components[:, 0],
            "PCA Component 1": components[:, 1]
        }

        plot = sns.scatterplot(data=data,
                               x="PCA Component 0",
                               y="PCA Component 1",
                               hue=labels,
                               style=labels,
                               s=100,
                               palette="tab10")
        plot.set_xlabel("PCA Component 0", fontsize=15)
        plot.set_ylabel("PCA Component 1", fontsize=15)
        plot.set_title("PCA of Trained Projections", fontsize=15)
        #plot.set_title("PCA of Untrained Projections", fontsize=15)

        plot.legend(loc="upper left", fontsize=12)
        #plot.legend_.remove()

        fig = plot.get_figure()
        fig.savefig(
            os.path.join(self.logdir, "combination-" + str(combination),
                         "pca.png"),
            bbox_inches='tight')
        print(filepaths)

    def make_arg_parser(self):
        super().make_arg_parser()
        self.parser.add_argument("-data_file")
        self.parser.add_argument("-num_authors")
        self.parser.add_argument("-num_files")

    def get_args(self):

        exp_type, exp, combination = super().get_args()

        self.data_file = self.args['data_file']
        self.num_authors = int(self.args['num_authors'])
        self.num_files = int(self.args['num_files'])

        return exp_type, exp, combination

    def output_hyperparameter_metrics(self, directory):

        pass


if __name__ == "__main__":
    AuthorPCA().execute()
