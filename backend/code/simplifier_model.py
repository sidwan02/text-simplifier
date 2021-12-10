import numpy as np
import tensorflow as tf
import transformer_components as transformer



class Simplifier_Transformer(tf.keras.Model):
    def __init__(self, complex_window_size, complex_vocab_size, simple_window_size, simple_vocab_size, hparams):

        ######vvv DO NOT CHANGE vvv##################
        super(Simplifier_Transformer, self).__init__()

        self.complex_vocab_size = complex_vocab_size  # The size of the complex vocab
        self.simple_vocab_size = simple_vocab_size  # The size of the simple vocab

        self.complex_window_size = complex_window_size  # The complex window size
        self.simple_window_size = simple_window_size  # The simple window size
        ######^^^ DO NOT CHANGE ^^^##################

        print(complex_vocab_size, simple_vocab_size)

        # Define batch size and optimizer/learning rate
        self.batch_size = 64
        self.embedding_size = hparams['embedding_size']
        # self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.optimizer = tf.keras.optimizers.Adam(hparams['adam_lr'])

        # Define simple and complex embedding layers:
        self.E_simple = tf.keras.layers.Embedding(
            self.simple_vocab_size, self.embedding_size)
        self.E_complex = tf.keras.layers.Embedding(
            self.complex_vocab_size, self.embedding_size)

        # Create positional encoder layers
        self.pos_enc_complex = transformer.Position_Encoding_Layer(
            self.complex_window_size, self.embedding_size)
        self.pos_enc_simple = transformer.Position_Encoding_Layer(
            self.simple_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.block_complex = transformer.Transformer_Block(
            self.embedding_size, is_decoder=False)
        self.block_simple = transformer.Transformer_Block(
            self.embedding_size, is_decoder=True)

        # Define dense layer(s)
        self.dense_1 = tf.keras.layers.Dense(80, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(120, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(200, activation="relu")
        self.dense_4 = tf.keras.layers.Dense(self.simple_vocab_size, activation="softmax")


    @tf.function
    def call(self, inputs):
        """
        :param encoder_input: batched ids corresponding to complex sentences
        :param decoder_input: batched ids corresponding to simple sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x simple_vocab_size]
        """

        encoder_input, decoder_input = inputs

        # 1) Add the positional embeddings to complex sentence embeddings
        embedding_complex = self.E_complex(encoder_input)

        # assert embedding_complex.shape == (self.batch_size, self.complex_window_size, self.embedding_size)

        rel_embeddings_complex = self.pos_enc_complex(embedding_complex)

        # 2) Pass the complex sentence embeddings to the encoder
        enc_out = self.block_complex(rel_embeddings_complex)

        # 3) Add positional embeddings to the simple sentence embeddings
        embedding_simple = self.E_simple(decoder_input)
        # assert embedding_simple.shape == (
        # self.batch_size, self.simple_window_size, self.embedding_size)

        rel_embeddings_simple = self.pos_enc_simple(embedding_simple)

        # 4) Pass the simple embeddings and output of your encoder, to the decoder
        dec_out = self.block_simple(rel_embeddings_simple, enc_out)

        # 5) Apply dense layer(s) to the decoder out to generate probabilities
        layer_1_out = self.dense_1(dec_out)
        layer_2_out = self.dense_2(layer_1_out)
        layer_3_out = self.dense_3(layer_2_out)
        probs = self.dense_4(layer_3_out)

        # assert probs.shape == (self.batch_size, self.simple_window_size, self.simple_vocab_size)

        return probs

    def train_step(self, data):
        train_complex_batch, train_simple_batch, labels = data

        # print("GOOGOO: ", train_complex_batch.shape)
        # print("GAGA: ", train_simple_batch.shape)
        

        with tf.GradientTape() as tape:
            probs = self((train_complex_batch, train_simple_batch))
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
        test_complex_batch, test_simple_batch, labels = data

        probs = self((test_complex_batch, test_simple_batch))
        loss = self.compiled_loss(y_true=labels, y_pred=probs)

        self.compiled_metrics.update_state(y_pred=probs, y_true=labels)
        return {m.name: m.result() for m in self.metrics}

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x simple_vocab_size]
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

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x simple_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        # assert prbs.shape == (
        # self.batch_size, self.simple_window_size, self.simple_vocab_size)
        # assert labels.shape == (self.batch_size, self.simple_window_size)
        # assert mask.shape == (self.batch_size, self.simple_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs), mask))