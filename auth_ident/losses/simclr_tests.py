from auth_ident.models.simclr_loss import SimCLRLoss
import tensorflow as tf
import math


def test_one_same_one_perpendicular():

    loss = SimCLRLoss(2, temperature=1.0)
    
    data = tf.math.l2_normalize([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], axis=1)
    loss = loss.call(None, data)
    print("Loss: ", loss)    
    print(math.isclose(loss, 1.266, rel_tol=1e-3))


def test_one_same_one_diff():

    loss = SimCLRLoss(2, temperature=1.0)
    
    data = tf.math.l2_normalize([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], axis=1)
    loss = loss.call(None, data)
    print("Loss:", loss)    
    print(math.isclose(loss, 1.3436, rel_tol=1e-4))


if __name__ == "__main__":

    test_one_same_one_perpendicular()
    test_one_same_one_diff()

