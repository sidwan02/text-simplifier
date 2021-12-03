import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys
import random
from tensorflow import keras

from attenvis import AttentionVis
av = AttentionVis()

import datetime

from tensorboard.plugins.hparams import api as hp

# Clear any logs from previous runs
# rm -rf ./logs/

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_log_dir = 'logs/fit/' + current_time
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

tensorboard_callback = keras.callbacks.TensorBoard(callback_log_dir)
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

# hparam_log_dir = 'logs/hparam_tuning' + current_time


# with tf.summary.create_file_writer(hparam_log_dir).as_default():
#   hp.hparams_config(
#     hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
#     metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
#   )

# hparam_callback = hp.KerasCallback(logdir, hparams)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()

    # Change this to "True" to turn on the attention matrix visualization.
    # You should turn this on once you feel your code is working.
    # Note that it is designed to work with transformers that have single attention heads.
    if sys.argv[1] == "TRANSFORMER":
        av.setup_visualization(enable=True)

    print("Running preprocessing...")
    train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data(
        '../../data/fls.txt', '../../data/els.txt', '../../data/flt.txt', '../../data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE, len(french_vocab),
                  ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args)

    # model.compile(optimizer=model.optimizer, run_eagerly=True)
    loss_per_symbol_metric = tf.keras.metrics.Mean(name="loss_per_symbol")
    acc_weighted_sum_metric = tf.keras.metrics.Mean(name="acc_weighted_sum")
    
    model.compile(optimizer=model.optimizer, metrics=[loss_per_symbol_metric, acc_weighted_sum_metric], run_eagerly=True)

    # ============

    train_french = train_french[:1000, :]

    train_english_trunc = train_english[:1000, :-1]

    labels = train_english[:1000, 1:]


    # ============
    # print("train_french.shape: ", train_french.shape)
    # print("train_english_trunc.shape: ", train_english_trunc.shape)
    # print("labels.shape: ", labels.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_french, train_english_trunc, labels))
    train_dataset = train_dataset.batch(model.batch_size)

    # model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])
    
    test_french = test_french[:1000, :]

    test_english_trunc = test_english[:1000, :-1]

    labels = test_english[:1000, 1:]

    test_dataset = tf.data.Dataset.from_tensor_slices((test_french, test_english_trunc, labels))
    test_dataset = test_dataset.batch(model.batch_size)

        

    
    # perplexity_metric = tf.keras.metrics.Mean(name="perplexity")

    model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback], validation_data=(test_dataset))

    # model.reset()


    # model.evaluate(test_dataset, callbacks=[tensorboard_callback])

    
    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    av.show_atten_heatmap()
    pass


if __name__ == '__main__':
    main()
