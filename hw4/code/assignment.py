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

from metrics import custom_loss, AccWeightedSum, Perplexity

# Clear any logs from previous runs
# rm -rf ./logs/

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# callback_log_dir = 'logs/fit/' + current_time
hparams_log_dir = 'logs/hparam_tuning/' + current_time
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=callback_log_dir, histogram_freq=1, update_freq='batch', embeddings_freq=1)
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))


# with tf.summary.create_file_writer(hparams_log_dir).as_default():
#   hp.hparams_config(
#     hparams=[HP_OPTIMIZER],
#     # https://stackoverflow.com/questions/56852300/hyperparameter-tuning-using-tensorboard-plugins-hparams-api-with-custom-loss-fun
#     metrics=[hp.Metric('AccWeightedSum', display_name='acc_weighted_sum'), hp.Metric('Perplexity', display_name='perplexity')],
#   )

# with tf.summary.create_file_writer(hparams_log_dir).as_default():
hp.hparams_config(
    hparams=[HP_OPTIMIZER],
    # https://stackoverflow.com/questions/56852300/hyperparameter-tuning-using-tensorboard-plugins-hparams-api-with-custom-loss-fun
    metrics=[hp.Metric('AccWeightedSum', display_name='acc_weighted_sum'), hp.Metric('Perplexity', display_name='perplexity')],
    )


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN", "TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()

    # Change this to "True" to turn on the attention matrix visualization.
    # You should turn this on once you feel your code is working.
    # Note that it is designed to work with transformers that have single attention heads.
    # if sys.argv[1] == "TRANSFORMER":
    av.setup_visualization(enable=True)

    print("Running preprocessing...")
    train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data(
        '../../data/fls.txt', '../../data/els.txt', '../../data/flt.txt', '../../data/elt.txt')
    print("Preprocessing complete.")

    print("eng_padding_index ======================: ", eng_padding_index)


    train_french = train_french[:1000, :]

    train_english_trunc = train_english[:1000, :-1]

    labels = train_english[:1000, 1:]

    # ============
    # print("train_french.shape: ", train_french.shape)
    # print("train_english_trunc.shape: ", train_english_trunc.shape)
    # print("labels.shape: ", labels.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_french, train_english_trunc, labels))
    train_dataset = train_dataset.batch(128)

    # model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])
    
    test_french = test_french[:1000, :]

    test_english_trunc = test_english[:1000, :-1]

    labels = test_english[:1000, 1:]

    test_dataset = tf.data.Dataset.from_tensor_slices((test_french, test_english_trunc, labels))
    test_dataset = test_dataset.batch(128)



    
    def run(hparams, logdir):
        print("hparams: ", hparams)

        model_args = (FRENCH_WINDOW_SIZE, len(french_vocab),
                    ENGLISH_WINDOW_SIZE, len(english_vocab), hparams)
        # if sys.argv[1] == "RNN":
        #     model = RNN_Seq2Seq(*model_args)
        # elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args)

        # model.compile(optimizer=model.optimizer, run_eagerly=True)
        # loss_per_symbol_metric = tf.keras.metrics.Mean(name="loss_per_symbol")
        # acc_weighted_sum_metric = tf.keras.metrics.Mean(name="acc_weighted_sum")
        
        model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)

        # ============
        # perplexity_metric = tf.keras.metrics.Mean(name="perplexity")

        model.fit(
            train_dataset, 
            epochs=3, 
            callbacks=[
                keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, update_freq='batch', embeddings_freq=1), 
                hp.KerasCallback(logdir, hparams)
                ], 
            validation_data=(test_dataset))

        # model.reset()


        # model.evaluate(test_dataset, callbacks=[tensorboard_callback])

        
        # Visualize a sample attention matrix from the test set
        # Only takes effect if you enabled visualizations above
        av.show_atten_heatmap()
        pass

    session_num = 0

    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            'optimizer': optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        run(hparams, hparams_log_dir + run_name)
        session_num += 1



if __name__ == '__main__':
    main()
