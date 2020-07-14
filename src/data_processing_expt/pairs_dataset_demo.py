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
    print("reading hdf...")
    df = pd.read_hdf('/home/spragunr/nobackup/pyfull.hdf')
    print("building generator...")
    pg = pairs_generator.PairGen(df, crop_length=1200, samples_per_epoch=1000)

    dataset = tf.data.Dataset.from_generator(
        pg.gen,
        ({"input_1": tf.string, "input_2": tf.string}, tf.bool))

    print(list(dataset.take(3).as_numpy_iterator()))

    print("timing...")
    benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
    #benchmark(dataset)
