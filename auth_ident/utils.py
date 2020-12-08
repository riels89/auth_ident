from tensorflow.keras import backend as K
from tensorflow.keras import Model
import tensorflow as tf
from auth_ident import param_mapping
import os


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def load_encoder(model, params, combination, logger, logdir):

    # Create inner model
    encoder = model.create_model(params, combination, logger)

    # Load most recent checkpoint
    logger.info(f"Encoder logdir: {logdir}")
    checkpoint_dir = os.path.join(logdir, f"combination-{combination}",
                                  "checkpoints")
    checkpoints = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        if f.endswith(".h5")
    ]
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

    encoder.load_weights(latest_checkpoint)

    return encoder

def get_data(params, dataset, k_nieghbors, data_file, return_file_indicies=False):

    print(params)
    if params['encoding_type'] == 'spm':
        dataset = dataset(crop_length=params["max_code_length"],
                          max_authors =params["max_authors"],
                          k_cross_val=k_nieghbors,
                          data_file=data_file,
                          encoding_type='spm',
                          spm_model_file=params['spm_model_file'])
    else:
        dataset = dataset(crop_length=params["max_code_length"],
                          max_authors = params["max_authors"],
                          k_cross_val=k_nieghbors,
                          data_file=data_file,
                          encoding_type=params['encoding_type'])

    params['dataset'] = dataset
    return dataset.get_dataset(return_file_indicies=return_file_indicies)
     

def get_model(contrastive_params,
              layer_name,
              normalize,
              combination,
              logger,
              logdir):
    print(contrastive_params)
    contrastive_model = param_mapping.map_model(contrastive_params)()
    encoder = load_encoder(contrastive_model, contrastive_params, combination, logger,
                           logdir)
    if normalize:
        output = tf.math.l2_normalize(
            encoder.get_layer(layer_name).output, axis=1)
    else:
        output = encoder.get_layer(layer_name).output

    base_model = Model(inputs=encoder.input[0],
                       outputs=output, name="base_model")

    base_model.summary()

    return base_model


def get_embeddings(params,
                   dataset,
                   max_authors,
                   k_cross_val,
                   output_layer_name,
                   data_file,
                   combination,
                   logger,
                   logdir,
                   normalize=True,
                   return_file_indicies=False):


    # Save as list to avoid extra if statement
    data = get_data(params, dataset, k_cross_val, data_file, return_file_indicies)
    X = data[0]
    y = data[1]

    embedding_model = get_model(params,
                                output_layer_name,
                                normalize,
                                combination,
                                logger,
                                logdir)

    embedding_model.compile(loss=lambda a, b, **kwargs: 0.0)
    embeddings = embedding_model.predict(X, batch_size=params["batch_size"])
    
    if return_file_indicies:
        return embeddings, y, data[2]
    else:
        return embeddings, y
