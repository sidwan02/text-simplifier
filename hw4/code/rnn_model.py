import numpy as np
import tensorflow as tf


class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        ###### DO NOT CHANGE ##############
        super(RNN_Seq2Seq, self).__init__()
        self.french_vocab_size = french_vocab_size  # The size of the french vocab
        self.english_vocab_size = english_vocab_size  # The size of the english vocab

        self.french_window_size = french_window_size  # The french window size
        self.english_window_size = english_window_size  # The english window size
        ######^^^ DO NOT CHANGE ^^^##################

        # TODO:
        # 1) Define any hyperparameters

        # Define batch size and optimizer/learning rate
        self.batch_size = 100  # You can change this
        self.embedding_size = 40  # You should change this
        self.optimizer = tf.keras.optimizers.Adam(0.001)

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.E_english = tf.keras.layers.Embedding(
            self.english_vocab_size, self.embedding_size)
        self.E_french = tf.keras.layers.Embedding(
            self.french_vocab_size, self.embedding_size)

        self.RNN_english = tf.keras.layers.GRU(
            50, return_sequences=True, return_state=True)
        self.RNN_french = tf.keras.layers.GRU(
            50, return_sequences=True, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(120, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(
            self.english_vocab_size, activation="softmax")

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # TODO:
        # 1) Pass your french sentence embeddings to your encoder
        embedding_french = self.E_french(encoder_input)
        whole_sequence_output_french, final_state_french = self.RNN_french(
            embedding_french, None)

        # 2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
        embedding_english = self.E_english(decoder_input)
        whole_sequence_output_english, final_state_english = self.RNN_english(
            embedding_english, final_state_french)

        # 3) Apply dense layer(s) to the decoder out to generate probabilities
        layer_1_out = self.dense_1(whole_sequence_output_english)
        layer_2_out = self.dense_2(layer_1_out)
        layer_3_out = self.dense_3(layer_2_out)
        probs = self.dense_4(layer_3_out)

        # assert probs.shape == (
        #     self.batch_size, self.english_window_size, self.english_vocab_size)

        return probs

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the total model cross-entropy loss after one forward pass. 
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # assert prbs.shape == (
        #     self.batch_size, self.english_window_size, self.english_vocab_size)
        # assert labels.shape == (self.batch_size, self.english_window_size)
        # assert mask.shape == (self.batch_size, self.english_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs), mask))
