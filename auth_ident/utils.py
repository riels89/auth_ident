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


def get_embeddings(params,
                   dataset,
                   k_cross_val,
                   data_file,
                   combination,
                   logger,
                   logdir,
                   return_file_indicies=False):

    dataset = dataset(crop_length=params["max_code_length"],
                      k_cross_val=k_cross_val,
                      data_file=data_file)
    params['dataset'] = dataset
    if return_file_indicies:
        data, labels, file_indicies = dataset.get_dataset(return_file_indicies=return_file_indicies)
    else:
        data, labels = dataset.get_dataset(return_file_indicies=return_file_indicies)

    contrastive_model = param_mapping.map_model(params)()
    encoder = load_encoder(contrastive_model, params, combination, logger,
                           logdir)

    layer_name = 'output_embedding'
    embedding_layer = Model(inputs=encoder.input[0],
                            outputs=tf.math.l2_normalize(
                                encoder.get_layer(layer_name).output, axis=1))
    embedding_layer.summary()

    embedding_layer.compile(loss=lambda a, b, **kwargs: 0.0)
    embeddings = embedding_layer.predict(data, batch_size=params["batch_size"])
    
    if return_file_indicies:
        return embeddings, labels, file_indicies
    else:
        return embeddings, labels
