import numpy as np
import tensorflow as tf
import transformer_components as transformer



class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		super(Transformer_Seq2Seq, self).__init__()

		# 1) Define hyperparameters
		self.window_size = _
		self.complex_vocab_size = _  #**** Think of way to deal with different vocab sizes for different lexile levels (unless we want to keep the vocab size constant at that of the highest lexile lv)
									# remember to adjust the __init__ parameters afterwards
		self.batch_size = _
		self.embedding_size = _
		self.learning_rate = _
		self.optimizer = _
		self.hidden_dense_size = _

		# 2) Define embeddings, encoder, decoder, and feed forward layers
		# Define complex and simple embedding layers:
		
		# Define encoder and decoder layers (2 encoder + 2 decoder blocks):

		# Define dense layer(s)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""	
		return None

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
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
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
		return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs) * mask)
	
