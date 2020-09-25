import tensorflow as tf
import numpy as np


class SimCLRLoss():

    def __init__(self, 
                 batch_size,
                 criterion=tf.nn.sparse_softmax_cross_entropy_with_logits,
                 temperature=1.0): 
        """
            Creates a keras loss to compute the contrastive loss
            for a batch.

            Args:
                
                batch_size (`int`):

                    The batch_size to assume. This allows some things to be
                    precomputed. This loss can still handel variable batch 
                    sizes, but if the actual batch size is different than the 
                    one provided it will be slower for that iteration.

                criterion (`tensorflow loss`, 
                           default=tf.nn.sparse_softmax_cross_entropy_with_logits):
                    
                    The loss function to use to compare the embs. Default is
                    tf.nn.sparse_softmax_cross_entropy_with_logits

                temperature (`int`):

                    The temperature to scale the distance by

        """
        self.criterion = criterion
        self.temperature = temperature
        self.batch_size = batch_size
        self.neg_mask = self._get_neg_mask(self.batch_size)

        bottom_half = tf.range(self.batch_size, 2 * self.batch_size)[np.newaxis]
        left_corner = tf.range(self.batch_size)[np.newaxis]

        self.index_l = tf.concat([bottom_half, left_corner], axis=0)
        self.index_l = tf.transpose(self.index_l, [1, 0])

        self.index_r = tf.concat([left_corner, bottom_half], axis=0)
        self.index_r = tf.transpose(self.index_r, [1, 0])

        self.__name__ = "simclr"

    def __call__(self, y_true, y_pred, **kwargs):
        """
            Computes the loss of the stacked contrastive embeddings.

            Args:

                y_true:
                    This does not matter, labels are computed automatically.
                    Just here to comply with keras loss.

                y_pred (`numpy array`):
                    The predictions of size (2 * batch, emb_size)

            Returns:
                The loss scalar
        """

        # METHOD 1
        # get actual batch_size and compare with given batch_size
        # if different recompute neg_mask
        b = int(y_pred.shape[0] / 2)
        if b != self.batch_size:
            curr_neg_mask = self._get_neg_mask(b)
        else:
            curr_neg_mask = self.neg_mask
        y_norm = tf.math.l2_normalize(y_pred, axis=1)

        sim = tf.matmul(y_norm, y_norm, transpose_b=True)
        pos = np.zeros((2 * b, 2 * b))

        pos_l = tf.gather_nd(sim, self.index_l)
        # TODO: We don't need to compute the righ side separately do we?
        pos_r = tf.gather_nd(sim, self.index_r)
        
        pos = tf.concat([pos_l, pos_r], axis=0)[:, np.newaxis]
        neg = tf.reshape(tf.boolean_mask(sim, curr_neg_mask), (2 * b, -1))

        logits = tf.concat([pos, neg], axis=1)
        logits /= self.temperature

        labels = tf.zeros(2 * b, dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.math.reduce_sum(loss) / float(2 * b)
        print("LOSS:", loss)

        # METHOD 2
        v1, v2 = tf.split(y_norm, 2, 0)

        labels = tf.range(b)
        masks = np.invert(np.identity(b, np.bool))

        # a = i and b = j, this makes it easier to read
        logits_aa = tf.matmul(v1, v1, transpose_b=True) / self.temperature
        logits_aa = tf.reshape(logits_aa[masks], (b, b - 1))

        logits_bb = tf.matmul(v2, v2, transpose_b=True) / self.temperature
        logits_bb = tf.reshape(logits_bb[masks], (b, b - 1))

        logits_ab = tf.matmul(v1, v2, transpose_b=True) / self.temperature
        # TODO: We don't need to compute the righ side separately do we?
        logits_ba = tf.matmul(v2, v1, transpose_b=True) / self.temperature
    
        loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b

        return tf.math.reduce_sum(loss) / float(2 * b)

    def _get_neg_mask(self, batch_size):
        """
        Computes the negative pair mask.

                F T F T
                T F T F
                F T F T
                T F T F

        This exculdes the simularity between an emb and itself (the main diagonal)
        and simularity between positive (matching) embs.
        """
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = l1 + l2 + diag 
        mask = (1 - mask).astype(bool)
        return mask 
