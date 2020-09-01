"""
Quick proof-of-concept illustration of a tf Dataset from a PairGen object.

Borrowed from:  https://www.tensorflow.org/guide/data_performance
"""
import pandas as pd
import tensorflow as tf
import pairs_generator
import src.preprocessing.split_dataset
import time

def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    time_sleeping = 0
    for epoch_num in range(num_epochs):
        for sample in dataset:
            time.sleep(0.0)
            time_sleeping += .0
            # Performing a training step


    tf.print("Execution time:", time.perf_counter() - start_time)
    tf.print("Time sleeping:", time_sleeping)


def test_simple():
    print("reading hdf...")
    df = pd.read_hdf('/home/spragunr/nobackup/pyfull.hdf')
    print("building generator...")
    pg = pairs_generator.PairGen(df, crop_length=1200, samples_per_epoch=1000)

    dataset = tf.data.Dataset.from_generator(
        pg.gen,
        ({"input_1": tf.uint8, "input_2": tf.uint8}, tf.bool))

    print(list(dataset.take(3).as_numpy_iterator()))

    print("timing...")
    benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
    #benchmark(dataset)

def test_full_pairs():
     sds = split_dataset.SplitDataset(1200, 64, language='java')
     train_dataset, val_dataset, test_dataset = sds.get_dataset()
     benchmark(train_dataset.prefetch(tf.data.experimental.AUTOTUNE))
     #benchmark(train_dataset)
     
if __name__ == "__main__":
    test_full_pairs()

