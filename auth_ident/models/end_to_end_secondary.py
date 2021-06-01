from auth_ident.models import GenericSecondaryClassifier 
from auth_ident.datasets import ClosedDataset
from auth_ident import param_mapping
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Softmax
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from math import ceil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from auth_ident.models import KNeighborSecondaryClassifier
#from auth_ident.optimizers import LearningRateMultiplier
import time
from tensorflow.keras.layers import Embedding, TimeDistributed, MaxPool1D
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, BatchNormalization, Lambda, Add


class EndToEndMLP(GenericSecondaryClassifier):
    """
    """
    def __init__(self, params, combination, logger, logdir):
        super().__init__(params, combination, logger, logdir)

        self.name = "end_to_end_mlp"
        self.dataset = ClosedDataset
        self.base_model = None
    
    def shuffle_and_batch(self, X_train, y_train):
        train_shuffle = np.random.permutation(np.arange(X_train.shape[0]))
        X_shuffled_train = X_train[train_shuffle]
        y_shuffled_train = y_train[train_shuffle]

        batch_size = self.params['model_params']['batch_size']
        train_num_splits = ceil(X_shuffled_train.shape[0]/ batch_size)
        X_batched_train = np.array_split(X_shuffled_train, train_num_splits)
        y_batched_train = np.array_split(y_shuffled_train, train_num_splits)

        return X_batched_train, y_batched_train

    def test_embeddings(self, X, y):
        #To make sure the embeddings work
        embeddings = self.base_model.predict(X, batch_size=256)
        knn_params =  {"k_cross_val": self.params["k_cross_val"], "model_params": {"n_neighbors": 1,"n_jobs": -1}}
        model = KNeighborSecondaryClassifier(knn_params, self.combination, self.logger, self.logdir)
        results = model.train(embeddings, y)
        print(f"Results: {results}")


    def make_CNN_model(self, input_shape, num_authors):
        """
        Implements CNN model from "Code authorship identification using convolutional neural networks"
        """
        param_mapping.map_params(self.params["model_params"])

        input = keras.Input(
            batch_shape=(None,
                         input_shape),
            name='secondary_model_input')

        embedding = Embedding(self.encoding_len,
                              256,
                              input_length=input_shape)(input)

        conv = Conv1D(128,
                      3,
                      strides=1,
                      padding="same",
                      activation="relu",
                      name='conv_1',
                      kernel_regularizer='l2')(embedding)
        conv = Dropout(0.6)(conv)
        conv = MaxPool1D(4, padding="valid", name="max_pool_1")(conv)

        conv = Conv1D(128,
                      5,
                      strides=1,
                      padding="same",
                      activation="relu",
                      name='conv_2',
                      kernel_regularizer='l2')(conv)
        conv = Dropout(0.6)(conv)
        conv = MaxPool1D(4, padding="valid", name="max_pool_2")(conv)

        conv = Conv1D(128,
                      7,
                      strides=1,
                      padding="same",
                      activation="relu",
                      name='conv_3',
                      kernel_regularizer='l2')(conv)
        conv = Dropout(0.6)(conv)
        conv = MaxPool1D(4, padding="valid", name="max_pool_3")(conv)
        conv = Flatten()(conv)

        prediction_layer = Dense(num_authors, name="prediction",
                      kernel_regularizer='l2')
        softmax = Softmax(name="prediction_probs")
        
        prediction = prediction_layer(conv)
        prediction_probs = softmax(prediction)
        
        model = keras.Model(inputs=input, outputs=prediction_probs)
        model.summary()
        return model

    def make_model(self, input_shape, num_authors):
        param_mapping.map_params(self.params["model_params"])

        input = keras.Input(
            batch_shape=(None,
                         input_shape),
            name='secondary_model_input')

        mlp1 = Dense(512,
                     activation=None,
                     name="MLP1",
                     kernel_regularizer='l1')
        mlp2 = Dense(512,
                     activation=None,
                     name="MLP2",
                     kernel_regularizer='l1')

        prediction_layer = Dense(num_authors, kernel_regularizer='l2', name="prediction")
        softmax = Softmax(name="prediction_probs")
        
        base = self.base_model(input) * 1
        mlp1_out = Dropout(rate=.5)(mlp1(base))
        mlp2_out = Dropout(rate=.5)(mlp2(mlp1_out))
        prediction = prediction_layer(mlp2_out)
        prediction_probs = softmax(prediction)
        
        model = keras.Model(inputs=input, outputs=prediction_probs)
        model.summary()
        return model 
    def train(self, X, y):

        assert self.base_model is not None, "Need to set base_model"
        
        num_authors = np.unique(y).shape[0]
        X_train = X[num_authors:]
        y_train = y[num_authors:]
        X_val = X[:num_authors]
        y_val = y[:num_authors]

        val_shuffle = np.random.permutation(np.arange(X_val.shape[0]))
        X_val = X_val[val_shuffle]
        y_val = y_val[val_shuffle]

        train_shuffle = np.random.permutation(np.arange(X_train.shape[0]))
        X_train = X_train[train_shuffle]
        y_train = y_train[train_shuffle]
        print(f"Number of files: {X_train.shape}")

        ##### Test Embeddings to make sure they work
        #self.test_embeddings(X, y)
        # Test inputing data as embeddings
        #X_train = self.base_model.predict(X_train, batch_size=256)
        #X_val = self.base_model.predict(X_val, batch_size=256)

        # Make self.make_CNN_model to test with other auth ident CNN
        model = self.make_model(X_train.shape[1], num_authors)
        
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        loss_fn = self.params['model_params']['loss']

        # Get weights of specific layers
        mlp_layers = ["MLP1", "MLP2", "prediction"]
        mlp_weights = [weight for layer in mlp_layers for weight in model.get_layer(layer).weights]
        base_weights = self.base_model.trainable_weights

        mlp_optimizer = self.params['model_params']['optimizer']
        lr = self.params['model_params']['lr']
        # Optimizer with different lr for base model
        base_optimizer = tf.keras.optimizers.Adam(learning_rate = lr * 0.0001) 

        X_batched_train, y_batched_train = self.shuffle_and_batch(X_train, y_train) 
        for epoch in range(self.params['model_params']['epochs']):
            X_batched_train, y_batched_train = self.shuffle_and_batch(
                np.concatenate(X_batched_train), np.concatenate(y_batched_train))
            loss_history = [] 
            for step, (x_batch, y_batch) in enumerate(zip(X_batched_train, y_batched_train)):
                with tf.GradientTape() as tape:

                    logits = model(x_batch, training=True)
                    loss_value = loss_fn(y_batch, logits) 

                    loss_history.append(loss_value)
                    train_acc_metric.update_state(y_batch, logits)

                # Get grads and separate into MLP and base model grads for their
                # separate optimziers
                grads = tape.gradient(loss_value, model.trainable_weights)
                mlp_grads = grads[len(self.base_model.trainable_weights):]
                base_grads = grads[:len(self.base_model.trainable_weights)]

                #mlp_optimizer.apply_gradients(zip(grads, model.trainable_weights))
                mlp_optimizer.apply_gradients(zip(mlp_grads, mlp_weights))
                base_optimizer.apply_gradients(zip(base_grads, base_weights))

            epoch_loss = sum(loss_history)  / len(X_batched_train)
            epoch_acc = train_acc_metric.result() 
            print(f"epoch {epoch} loss: {epoch_loss}")
            print(f"epoch {epoch} acc: {epoch_acc}")

            val_logits = model.predict(X_val) 
            val_loss = loss_fn(y_val, val_logits)

            val_acc_metric.update_state(y_val, val_logits)
            val_acc = val_acc_metric.result()
            print(f"val loss: {val_loss}")
            print(f"val acc: {val_acc}")

            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

        val_logits = model.predict(X_val) 
        val_loss = loss_fn(y_val, val_logits)

        val_acc_metric.update_state(y_val, val_logits)
        val_acc = val_acc_metric.result()
        print(f"FINAL val loss: {val_loss}")
        print(f"FINAL val acc: {val_acc}")

        return val_acc

    def set_base_model(self, base_model):
        self.base_model = base_model

    def set_encoding_len(self, encoding_len):
        self.encoding_len = encoding_len

    def evaluate(self, X, y=None): 

        if y is None:
            return self.model.predict(X) 
        else:
            return self.model.score(X, y)

    def save():
        pass
