"""
Quick proof-of-concept illustration of a tf Dataset from a PairGen object.

Borrowed from:  https://www.tensorflow.org/guide/data_performance
"""
import pandas as pd
import tensorflow as tf
import pairs_generator
import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    time_sleeping = 0
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
            time_sleeping += .01


    tf.print("Execution time:", time.perf_counter() - start_time)
    tf.print("Time sleeping:", time_sleeping)


if __name__ == "__main__":
    df = pd.read_hdf('/home/spragunr/auth_ident/py.hdf')
    pg = pairs_generator.PairGen(df, crop_length=10, samples_per_epoch=100)

    dataset = tf.data.Dataset.from_generator(
        pg.gen,
        (tf.string, tf.string, tf.int32))

    print(list(dataset.take(3).as_numpy_iterator()))

    #benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
    benchmark(dataset)
