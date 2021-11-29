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

# Clear any logs from previous runs
# rm -rf ./logs/

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_log_dir = 'logs/fit/' + current_time
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

tensorboard_callback = keras.callbacks.TensorBoard(callback_log_dir)
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

def train(model, train_french, train_english, eng_padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """

    num_sentences = len(train_french)

    # print("train_french.shape: ", train_french.shape)
    # print("train_english.shape: ", train_english.shape)

    # assert train_french.shape == (num_sentences, 14)
    # assert train_english.shape == (num_sentences, 15)

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder,
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
    #
    # - When computing loss, the decoder labels should have the first word removed:
    #	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

    batch_num = 0
    for i in range(0, len(train_french) - model.batch_size, model.batch_size):
        batch_num += 1

        train_french_batch = train_french[i: i + model.batch_size, :]

        # input of decoder
        train_english_batch = train_english[i: i + model.batch_size, :-1]
        # somehow create a mask from the padding index
        # for now, make this a dummy matrix of 1s

        labels = train_english[i: i + model.batch_size, 1:]  # labels

        model.train_step((train_french_batch, train_english_batch, labels, eng_padding_index))


@av.test_func
def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
    e.g. (my_perplexity, my_accuracy)
    """

    # Note: Follow the same procedure as in train() to construct batches of data!

    total_valid_tokens = 0
    acc_loss = 0
    weighted_sum_acc = 0

    batch_num = 0
    for i in range(0, len(test_french) - model.batch_size, model.batch_size):
        batch_num += 1
        test_french_batch = test_french[i: i + model.batch_size, :]

        # somehow create a mask from the padding index
        # for now, make this a dummy matrix of 1s
        mask = []

        # this is used later for the perp per symbol and acc per symbol calculations
        batch_valid_tokens = 0

        test_english_batch = test_english[i: i + model.batch_size, :-1]

        labels = test_english[i: i + model.batch_size, 1:]

        for indexed_sentence in labels:
            sentence_mask = [
                0 if pad_index == eng_padding_index else 1 for pad_index in indexed_sentence]
            mask.append(sentence_mask)
            batch_valid_tokens += np.sum(sentence_mask)

        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        total_valid_tokens += batch_valid_tokens

        with tf.GradientTape() as tape:
            probs = model(test_french_batch, test_english_batch)
            loss = model.loss_function(probs, labels, mask)

            acc_loss += loss

            acc = model.accuracy_function(probs, labels, mask)
            weighted_sum_acc += acc * batch_valid_tokens

        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=batch_num)
        #     tf.summary.scalar('accuracy', acc, step=batch_num)

    perplexity = tf.exp(acc_loss / total_valid_tokens)
    accuracy = weighted_sum_acc / total_valid_tokens

    return perplexity, accuracy


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

    model.compile(optimizer=model.optimizer, run_eagerly=True)

    # ============

    train_french = train_french[:, :]

    train_english_trunc = train_english[:, :-1]

    labels = train_english[:, 1:]


    # ============
    print("train_french.shape: ", train_french.shape)
    print("train_english_trunc.shape: ", train_english_trunc.shape)
    print("labels.shape: ", labels.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_french, train_english_trunc, labels))
    train_dataset = train_dataset.batch(64)

    model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback])

    # TODO:
    # Train and Test Model for 1 epoch.
    # for num_epoch in range(1, 2, 1):
    #     for batch_num, data in enumerate(train_dataset):
    #         model.train_step(data, batch_num)

        # train(model, train_french, train_english, eng_padding_index)

    # perplexity, accuracy = test(
    #     model, test_french, test_english, eng_padding_index)
    # print("perplexity per symbol: ", perplexity)
    # print("accuracy per symbol: ", accuracy)

    # Visualize a sample attention matrix from the test set
    # Only takes effect if you enabled visualizations above
    av.show_atten_heatmap()
    pass


if __name__ == '__main__':
    main()
