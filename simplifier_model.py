import numpy as np
import tensorflow as tf
import transformer_components as transformer



class Simplifier_Transformer(tf.keras.Model):
    def __init__(self, lexile_high_window_size, lexile_high_vocab_size, lexile_low_window_size, lexile_low_vocab_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Simplifier_Transformer, self).__init__()

        self.lexile_high_vocab_size = lexile_high_vocab_size  # The size of the lexile_high vocab
        self.lexile_low_vocab_size = lexile_low_vocab_size  # The size of the lexile_low vocab

        self.lexile_high_window_size = lexile_high_window_size  # The lexile_high window size
        self.lexile_low_window_size = lexile_low_window_size  # The lexile_low window size
        ######^^^ DO NOT CHANGE ^^^##################

        print(lexile_high_vocab_size, lexile_low_vocab_size)

        # Define batch size and optimizer/learning rate
        self.batch_size = 64
        self.embedding_size = 50
        # self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.optimizer = tf.keras.optimizers.Adam(0.001)

        # Define lexile_low and lexile_high embedding layers:
        self.E_lexile_low = tf.keras.layers.Embedding(
            self.lexile_low_vocab_size, self.embedding_size)
        self.E_lexile_high = tf.keras.layers.Embedding(
            self.lexile_high_vocab_size, self.embedding_size)

        # Create positional encoder layers
        self.pos_enc_lexile_high = transformer.Position_Encoding_Layer(
            self.lexile_high_window_size, self.embedding_size)
        self.pos_enc_lexile_low = transformer.Position_Encoding_Layer(
            self.lexile_low_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.block_lexile_high = transformer.Transformer_Block(
            self.embedding_size, is_decoder=False)
        self.block_lexile_low = transformer.Transformer_Block(
            self.embedding_size, is_decoder=True)

        # Define dense layer(s)
        self.dense_1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(120, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(self.lexile_low_vocab_size, activation="softmax")


    @tf.function
    def call(self, inputs):
        """
        :param encoder_input: batched ids corresponding to lexile_high sentences
        :param decoder_input: batched ids corresponding to lexile_low sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x lexile_low_vocab_size]
        """

        encoder_input, decoder_input = inputs

        # 1) Add the positional embeddings to lexile_high sentence embeddings
        embedding_lexile_high = self.E_lexile_high(encoder_input)

        # assert embedding_lexile_high.shape == (self.batch_size, self.lexile_high_window_size, self.embedding_size)

        rel_embeddings_lexile_high = self.pos_enc_lexile_high(embedding_lexile_high)

        # 2) Pass the lexile_high sentence embeddings to the encoder
        enc_out = self.block_lexile_high(rel_embeddings_lexile_high)

        # 3) Add positional embeddings to the lexile_low sentence embeddings
        embedding_lexile_low = self.E_lexile_low(decoder_input)
        # assert embedding_lexile_low.shape == (
        # self.batch_size, self.lexile_low_window_size, self.embedding_size)

        rel_embeddings_lexile_low = self.pos_enc_lexile_low(embedding_lexile_low)

        # 4) Pass the lexile_low embeddings and output of your encoder, to the decoder
        dec_out = self.block_lexile_low(rel_embeddings_lexile_low, enc_out)

        # 5) Apply dense layer(s) to the decoder out to generate probabilities
        layer_1_out = self.dense_1(dec_out)
        layer_2_out = self.dense_2(layer_1_out)
        layer_3_out = self.dense_3(layer_2_out)
        probs = self.dense_4(layer_3_out)

        # assert probs.shape == (self.batch_size, self.lexile_low_window_size, self.lexile_low_vocab_size)

        return probs

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x lexile_low_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        labels = tf.cast(labels, dtype=tf.int64)

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x lexile_low_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        # assert prbs.shape == (
        # self.batch_size, self.lexile_low_window_size, self.lexile_low_vocab_size)
        # assert labels.shape == (self.batch_size, self.lexile_low_window_size)
        # assert mask.shape == (self.batch_size, self.lexile_low_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs), mask))
	
