from auth_ident.datasets import ClosedDataset
from auth_ident import GenericExecute, param_mapping
from auth_ident.utils import get_model, get_data
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from os.path import join
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
import itertools


class DensityPlot(GenericExecute):

    def get_embeddings(self, model, max_code_length):

        f = join("data/", self.data_file)
        embedding_dataframe = pd.read_hdf(f)
        files_by_auth_name = embedding_dataframe.groupby(['username']).indices

        authors_with_2 = dict(list(dict(
            filter(lambda x: len(x[1]) >= 2,
                files_by_auth_name.items())).items())[:1500])

        authors_with_2_embeddings = {}
        for author, file_indicies in authors_with_2.items():
            file_content = embedding_dataframe['file_content'][file_indicies].values
            padded_content = np.empty([len(file_content), max_code_length])
            for index, file in enumerate(file_content):
                padded_content[index, :len(file)] = file[:max_code_length]
                padded_content[index, len(file):] = 0

            authors_with_2_embeddings[author] = model.predict(padded_content, padded_content.shape[0])

        return authors_with_2_embeddings

    def execute_one(self, contrastive_params, combination, logger):
        embedding_size = contrastive_params["embedding_size"]
        get_data(contrastive_params, 
                       ClosedDataset,
                       k_nieghbors=2,
                       data_file=self.data_file,
                       return_file_indicies=False)
        

        model = get_model(contrastive_params,
                          layer_name="output_embedding",
                          normalize=True,
                          combination=combination,
                          logger=logger,
                          logdir=self.logdir)
        max_code_length = contrastive_params['max_code_length']
        author_embeddings = self.get_embeddings(model, max_code_length)
        author_embeddings_list = np.array([embedding for v in author_embeddings.values() for embedding in v])

        same_auth_sim = []
        diff_auth_sim = []

        rng = np.random.default_rng(1)
        curr_embedding_list_index = 0
        for index, (author, embeddings) in tqdm(enumerate(author_embeddings.items())):
            same_sim_matrix = cosine_similarity(embeddings, embeddings)
            same_sims = same_sim_matrix[np.triu_indices(same_sim_matrix.shape[0], k=1)].flatten()
            same_auth_sim.append(list(same_sims))
            
            other_author_embeddings = np.concatenate([ 
                author_embeddings_list[:curr_embedding_list_index].reshape([-1, embedding_size]),
                author_embeddings_list[curr_embedding_list_index + embeddings.shape[0]:].reshape([-1, embedding_size])])
            other_author_embeddings = rng.choice(other_author_embeddings,
                                            50,
                                            replace=False)
            diff_sim_matrix = cosine_similarity(other_author_embeddings, embeddings).flatten()
            #diff_sims = diff_sim_matrix[np.triu_indices(diff_sim_matrix.shape[0], k=1)].flatten()
            diff_auth_sim.append(diff_sim_matrix)

            curr_embedding_list_index += embeddings.shape[0]
        print(len(same_auth_sim))
        print(len(diff_auth_sim))
        same_auth_sim = list(itertools.chain.from_iterable(same_auth_sim))
        diff_auth_sim = list(itertools.chain.from_iterable(diff_auth_sim))
        print(len(same_auth_sim))
        print(len(diff_auth_sim))
        print(np.amax(same_auth_sim))
        print(np.amin(same_auth_sim))
        print(np.amax(diff_auth_sim))
        print(np.amin(diff_auth_sim))

        sim_type = np.concatenate([np.ones(len(same_auth_sim)), np.zeros(len(diff_auth_sim))])
        sim = same_auth_sim + diff_auth_sim
        data = pd.DataFrame({"Similarity": sim, "sim_type": sim_type})
        #plt.rcParams.update(**{'axes.labelsize':50})
        sns.set_style("ticks", {"xtick.major.size": 40, "ytick.major.size": 40})
        plot = sns.kdeplot(data=data[data['sim_type'] == 0],
                               x="Similarity",
                               color='b',
                               fill=True,
                               bw_adjust=.25)
        plot = sns.kdeplot(data=data[data['sim_type'] == 1],
                               x="Similarity",
                               color='r',
                               fill=True,
                               bw_adjust=.25)
        #plot.set_xlabel("PCA Component 0", fontsize=15)
        #plot.set_ylabel("PCA Component 1", fontsize=15)
        #plot.set_title("PCA of Trained Projections", fontsize=15)
        #plot.set_title("PCA of Untrained Projections", fontsize=15)

        plot.legend(labels = ["Different Authors", "Same Authors"], loc="upper left", fontsize=15)
        plot.set_xlabel("Similarity", fontsize=15)
        plot.set_ylabel("Density", fontsize=15)
        plot.set_xticklabels(['{0:.2f}'.format(tick) for tick in plot.get_xticks()], fontsize=14)
        plot.set_yticklabels(['{0:.2f}'.format(tick) for tick in plot.get_yticks()], fontsize=14)
        #plot.legend_.remove()

        fig = plot.get_figure()
        fig.savefig(
            os.path.join(self.logdir, "combination-" + str(combination),
                         "similarity_dist.png"),
            bbox_inches='tight')

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
    DensityPlot().execute()
