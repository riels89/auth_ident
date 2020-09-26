from auth_ident import GenericExecute, param_mapping
from auth_ident.utils import get_embeddings
from sklearn.decomposition import PCA
import seaborn as sns
import os
import numpy as np


class AuthorPCA(GenericExecute):
    def execute_one(self, contrastive_params, combination, logger):

        self.model = param_mapping.map_model(contrastive_params)(
            contrastive_params, combination, logger)

        train_data, train_labels = get_embeddings(contrastive_params,
                                                  self.model.dataset,
                                                  self.num_files,
                                                  data_file=self.data_file,
                                                  combination=combination,
                                                  logger=logger,
                                                  logdir=self.logdir)

        authors_per_split = int(train_data.shape[0] / float(self.num_files))
        files = np.empty(
            (self.num_authors * self.num_files, train_data.shape[1]))
        labels = np.empty((self.num_authors * self.num_files), dtype=np.int16)
        for i in range(self.num_authors):
            files[i * self.num_authors:i * self.num_authors +
                  self.num_authors] = train_data[i * authors_per_split:self.
                                                 num_authors +
                                                 i * authors_per_split]

            labels[i * self.num_authors:self.num_authors * (i + 1)] = \
                train_labels[i * authors_per_split:self.num_authors + i
                             * authors_per_split]
        pca = PCA(n_components=train_data.shape[1])
        components = pca.fit_transform(files)
        print(f"explained variance: {pca.explained_variance_ratio_}")

        data = {
            "component_0": components[:, 0],
            "component_1": components[:, 1]
        }

        plot = sns.scatterplot(data=data,
                               x="component_0",
                               y="component_1",
                               hue=labels,
                               palette="tab10")
        fig = plot.get_figure()
        fig.savefig(
            os.path.join(self.logdir, "combination-" + str(combination),
                         "pca.png"))

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

    def output_hypeparameter_metrics(self, directory):

        pass


if __name__ == "__main__":
    AuthorPCA().execute()
