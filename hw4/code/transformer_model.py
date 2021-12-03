import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()



# loss_per_symbol_metric = tf.keras.metrics.Mean(name="loss_per_symbol")
# acc_weighted_sum_metric = tf.keras.metrics.Mean(name="acc_weighted_sum")
# perplexity_metric = tf.keras.metrics.Mean(name="perplexity")
# mae_metric = tf.keras.metrics.Accuracy(name="accuracy")

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

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
        self.optimizer = tf.keras.optimizers.Adam(0.001)

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

        self.weighted_sum_acc = 0
        self.total_valid_tokens = 0
        self.acc_loss = 0

        self.is_training = True

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
        # print("in train_step ===========================================")
        eng_padding_index = 0
        # train_french_batch, train_english_batch, labels, eng_padding_index, test_summary_writer, batch_num = data
        train_french_batch, train_english_batch, labels = data




        batch_valid_tokens = 0

        # this is used later for the perp per symbol and acc per symbol calculations

        # print(labels)



        # mask = []
        # for indexed_sentence in labels.numpy():
        #     sentence_mask = [
        #         0 if pad_index == eng_padding_index else 1 for pad_index in indexed_sentence]
        #     mask.append(sentence_mask)
        #     batch_valid_tokens += np.sum(sentence_mask)

        # mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        mask = tf.where(labels == eng_padding_index, 0, 1)

        batch_valid_tokens += tf.cast(tf.math.reduce_sum(mask), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            probs = self((train_french_batch, train_english_batch))
            loss = self.loss_function(probs, labels, mask)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(probs, labels, mask)

        # loss_per_symbol_metric.update_state(loss)
        # mae_metric.update_state(probs, labels)

        self.total_valid_tokens += batch_valid_tokens
        # self.acc_loss += loss

        acc = self.accuracy_function(probs, labels, mask)
        # self.weighted_sum_acc += acc * batch_valid_tokens

        perplexity = tf.exp(self.acc_loss / self.total_valid_tokens)
        # accuracy = self.weighted_sum_acc / self.total_valid_tokens

        if len(self.metrics) >= 1:
            self.metrics[0].update_state(loss / batch_valid_tokens)
        if len(self.metrics) >= 2:
            self.metrics[1].update_state(acc, sample_weight=batch_valid_tokens)

        # loss_per_symbol_metric.update_state(loss / batch_valid_tokens)
        # acc_weighted_sum_metric.update_state(acc, sample_weight=batch_valid_tokens)
        # perplexity_metric.update_state(perplexity)

        # return {"loss per symbol": loss_per_symb, "accuracy": accuracy, "perplexity": perplexity}
        # return {"loss per symbol": loss_per_symbol_metric.result(), "accuracy": acc_weighted_sum_metric.result(), "perplexity": perplexity_metric.result()}

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=batch_num)

        # print("Loss per symbol: {}".format(loss / batch_valid_tokens))

    def reset(self):
        self.weighted_sum_acc = 0
        self.total_valid_tokens = 0
        self.acc_loss = 0

    def test_step(self, data):
        if self.is_training:
            self.reset()
            self.is_training = False

        # print("in train_step ===========================================")
        eng_padding_index = 0
        # train_french_batch, train_english_batch, labels, eng_padding_index, test_summary_writer, batch_num = data
        train_french_batch, train_english_batch, labels = data


        batch_valid_tokens = 0

        # this is used later for the perp per symbol and acc per symbol calculations

        # print(labels)

        mask = []
        for indexed_sentence in labels.numpy():
            sentence_mask = [
                0 if pad_index == eng_padding_index else 1 for pad_index in indexed_sentence]
            mask.append(sentence_mask)
            batch_valid_tokens += np.sum(sentence_mask)

        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        probs = self((train_french_batch, train_english_batch))
        loss = self.loss_function(probs, labels, mask)

        self.total_valid_tokens += batch_valid_tokens
        # self.acc_loss += loss

        acc = self.accuracy_function(probs, labels, mask)
        # self.weighted_sum_acc += acc * batch_valid_tokens

        perplexity = tf.exp(self.acc_loss / self.total_valid_tokens)
        # accuracy = self.weighted_sum_acc / self.total_valid_tokens


        # loss_per_symbol_metric.update_state(loss / batch_valid_tokens)
        # acc_weighted_sum_metric.update_state(acc, sample_weight=batch_valid_tokens)

        if len(self.metrics) >= 1:
            self.metrics[0].update_state(loss / batch_valid_tokens)
        if len(self.metrics) >= 2:
            self.metrics[1].update_state(acc, sample_weight=batch_valid_tokens)

        return {m.name: m.result() for m in self.metrics}

        # return {"loss per symbol": loss_per_symb, "accuracy": accuracy, "perplexity": perplexity}
        # return {"loss per symbol": loss_per_symbol_metric.result(), "accuracy": acc_weighted_sum_metric.result()}

    # @property
    # def metrics(self):
    #     # We list our `Metric` objects here so that `reset_states()` can be
    #     # called automatically at the start of each epoch
    #     # or at the start of `evaluate()`.
    #     # If you don't implement this property, you have to call
    #     # `reset_states()` yourself at the time of your choosing.
    #     return [loss_per_symbol_metric, acc_weighted_sum_metric, perplexity_metric]

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

    @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
