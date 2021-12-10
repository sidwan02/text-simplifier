import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

# from attenvis import AttentionVis

# av = AttentionVis()



# loss_per_symbol_metric = tf.keras.metrics.Mean(name="loss_per_symbol")
# acc_weighted_sum_metric = tf.keras.metrics.Mean(name="acc_weighted_sum")
# perplexity_metric = tf.keras.metrics.Mean(name="perplexity")
# mae_metric = tf.keras.metrics.Accuracy(name="accuracy")

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size, hparams):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.french_vocab_size = french_vocab_size  # The size of the french vocab
        self.english_vocab_size = english_vocab_size  # The size of the english vocab

        self.french_window_size = french_window_size  # The french window size
        self.english_window_size = english_window_size  # The english window size
        ######^^^ DO NOT CHANGE ^^^##################

        # TODO:
        # 1) Define any hyperparameters
        # 2) Define embeddings, encoder, decoder, and feed forward layers

        # Define batch size and optimizer/learning rate
        self.batch_size = 128
        self.embedding_size = 40
        # self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.optimizer = tf.keras.optimizers.Adam(hparams['adam_lr'])

        # Define english and french embedding layers:
        self.E_english = tf.keras.layers.Embedding(
            self.english_vocab_size, self.embedding_size)
        self.E_french = tf.keras.layers.Embedding(
            self.french_vocab_size, self.embedding_size)

        # Create positional encoder layers
        self.pos_enc_french = transformer.Position_Encoding_Layer(
            self.french_window_size, self.embedding_size)
        self.pos_enc_english = transformer.Position_Encoding_Layer(
            self.english_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.block_french = transformer.Transformer_Block(
            self.embedding_size, is_decoder=False)
        self.block_english = transformer.Transformer_Block(
            self.embedding_size, is_decoder=True)

        # Define dense layer(s)
        self.dense_1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(120, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(
            self.english_vocab_size, activation="softmax")


    @tf.function
    def call(self, inputs):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        encoder_input, decoder_input = inputs

        # TODO:
        # 1) Add the positional embeddings to french sentence embeddings
        embedding_french = self.E_french(encoder_input)

        # assert embedding_french.shape == (self.batch_size, self.french_window_size, self.embedding_size)

        rel_embeddings_french = self.pos_enc_french(embedding_french)

        # 2) Pass the french sentence embeddings to the encoder
        enc_out = self.block_french(rel_embeddings_french)

        # 3) Add positional embeddings to the english sentence embeddings
        embedding_english = self.E_english(decoder_input)
        # assert embedding_english.shape == (
        # self.batch_size, self.english_window_size, self.embedding_size)

        rel_embeddings_english = self.pos_enc_english(embedding_english)

        # 4) Pass the english embeddings and output of your encoder, to the decoder
        dec_out = self.block_english(rel_embeddings_english, enc_out)

        # 5) Apply dense layer(s) to the decoder out to generate probabilities
        layer_1_out = self.dense_1(dec_out)
        layer_2_out = self.dense_2(layer_1_out)
        layer_3_out = self.dense_3(layer_2_out)
        probs = self.dense_4(layer_3_out)

        # assert probs.shape == (self.batch_size, self.english_window_size, self.english_vocab_size)

        return probs

    def train_step(self, data):
        train_french_batch, train_english_batch, labels = data

        # print("GOOGOO: ", train_french_batch.shape)
        # print("GAGA: ", train_english_batch.shape)
        

        with tf.GradientTape() as tape:
            probs = self((train_french_batch, train_english_batch))
            loss =  self.compiled_loss(y_true=labels, y_pred=probs)

        # print("probs: ", probs)
        # print("loss: ---", loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_pred=probs, y_true=labels)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        test_french_batch, test_english_batch, labels = data

        probs = self((test_french_batch, test_english_batch))
        loss = self.compiled_loss(y_true=labels, y_pred=probs)

        self.compiled_metrics.update_state(y_pred=probs, y_true=labels)
        return {m.name: m.result() for m in self.metrics}

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
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

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        # assert prbs.shape == (
        # self.batch_size, self.english_window_size, self.english_vocab_size)
        # assert labels.shape == (self.batch_size, self.english_window_size)
        # assert mask.shape == (self.batch_size, self.english_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs), mask))

    # @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
