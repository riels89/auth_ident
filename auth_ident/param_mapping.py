from tensorflow import keras
from auth_ident.losses import SimCLRLoss, ContrastiveMarginLoss
from auth_ident import models
from auth_ident import datasets
from itertools import product


def generate_param_grid(params):
    return [
        dict(zip(params.keys(), values))
        for values in product(*params.values())
    ]


def map_model(params):
    return models.model_map[params['model']]


def map_dataset(dataset_type, params, data_file):

    if dataset_type == "combined":
        dataset = datasets.CombinedDataset(
            max_code_length=params["max_code_length"],
            batch_size=params['batch_size'],
            binary_encoding=params['binary_encoding'])

    elif dataset_type == "split":
        if params['loss'] == 'margin':
            dataset = datasets.SplitDataset(
                max_code_length=params["max_code_length"],
                batch_size=params['batch_size'],
                binary_encoding=params['binary_encoding'],
                data_file=data_file)

        elif params['loss'] == "simclr":
            if "spm_model_file" in params:
                dataset = datasets.SimCLRDataset(
                    max_code_length=params["max_code_length"],
                    batch_size=params['batch_size'],
                    data_file=data_file,
                    encoding_type=params['encoding_type'],
                    spm_model_file=params['spm_model_file'])
            else:
                dataset = datasets.SimCLRDataset(
                    max_code_length=params["max_code_length"],
                    batch_size=params['batch_size'],
                    data_file=data_file,
                    encoding_type=params['encoding_type'])

    elif dataset_type == 'by_line':
        if params['loss'] == 'margin':
            dataset = datasets.ByLineDataset(
                max_lines=params["max_lines"],
                max_line_length=params["max_line_length"],
                batch_size=params['batch_size'],
                binary_encoding=params['binary_encoding'])
        else:
            dataset = datasets.ByLineDataset(
                max_lines=params["max_lines"],
                max_line_length=params["max_line_length"],
                batch_size=params['batch_size'],
                binary_encoding=params['binary_encoding'])

    params['dataset'] = dataset

    return dataset.create_dataset()


def map_params(params):
    if params['optimizer'] == 'adam':
        kwargs = {}
        if 'lr' in params:
            kwargs['lr'] = params['lr']
        if 'clipvalue' in params:
            kwargs['clipvalue'] = params['clipvalue']
        elif 'clipnorm' in params:
            kwargs['clipnorm'] = params['clipnorm']
        if 'decay' in params:
            kwargs['decay'] = params['decay']
        params['optimizer'] = keras.optimizers.Adam(**kwargs)

    if params['loss'] == 'contrastive':
        margin = 1.0
        if 'margin' in params:
            margin = params['margin']
        params['loss'] = ContrastiveMarginLoss(margin)

    if params['loss'] == 'simclr':
        # Will throw error if temp not specified, this is wanted
        params['loss'] = SimCLRLoss(
            params['batch_size'],
            temperature=params['temperature'])
