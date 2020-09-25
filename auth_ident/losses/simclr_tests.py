from auth_ident.losses.simclr_loss import SimCLRLoss
import tensorflow as tf
import math


def test_one_same_one_perpendicular():

    loss = SimCLRLoss(2, temperature=1.0)

    data = tf.convert_to_tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=tf.float32)
    loss = loss.__call__(None, data)
    print("Loss: ", loss)    
    print(math.isclose(loss, 1.266, rel_tol=1e-3))


def test_one_same_one_diff():

    loss = SimCLRLoss(2, temperature=1.0)

    data = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32)

    loss = loss.__call__(None, data)
    print("Loss:", loss)    
    print(math.isclose(loss, 1.3436, rel_tol=1e-4))


if __name__ == "__main__":

    test_one_same_one_perpendicular()
    test_one_same_one_diff()

