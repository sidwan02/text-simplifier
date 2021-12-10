import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from simplifier_model import Simplifier_Transformer
import sys
import random
import re
from tensorflow import keras


def simplify_main(text_input, simplification_strength=1):
    
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	data_root = os.path.dirname(cur_dir) + '/data'

	UNK_TOKEN = "*UNK*"
	print("Running preprocessing...")
	train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data(data_root+'/wiki_normal_train.txt', data_root+'/wiki_simple_train.txt', data_root+'/wiki_normal_test.txt', data_root+'/wiki_simple_test.txt')
	# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data('./dummy_data/wiki_normal_train.txt','./dummy_data/wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
	# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/fls.txt','./dummy_data/els.txt','./dummy_data/flt.txt','./dummy_data/elt.txt')
	vocab_word_list = list(simple_vocab.keys())
	vocab_idx_list = list(simple_vocab.values())
	stop_complex_id = complex_vocab["*STOP*"]
	pad_complex_id = complex_vocab["*PAD*"]
	start_simple_id = simple_vocab["*START*"]
	stop_simple_id = simple_vocab["*STOP*"]
	print("Preprocessing complete.")

	# ====

	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
	model = Simplifier_Transformer(*model_args)
	
	model((np.zeros((64, 420)), np.zeros((64, 220))))
	
	model.load_weights(cur_dir + "/model.h5")
	
	model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)

	def call_inference(model, input_ids):
		"""
		Used for inference without ground truth labels. Works by calling call() recursively,
		using the decoder output of the previous iteration each time (starting with the <START> token)
		as the new decoder input
		:param input_ids: 1D tensor of word ids of the input text string, [sentence_length,]
		:return: full string corresponding the the input_ids
		"""
		# encoder_input = np.reshape(input_ids[:SIMPLE_WINDOW_SIZE], [1, len(input_ids)]) #reshape to [1, sent_len]
		accumulated_output = [] # first word to be generated is after start token

		# pad encoder input so that it is [COMPLEX_WINDOW_SIZE,]
		encoder_input = input_ids[:COMPLEX_WINDOW_SIZE]
		encoder_input = encoder_input + [stop_complex_id] + [pad_complex_id] * (COMPLEX_WINDOW_SIZE - len(encoder_input)-1)
		encoder_input = tf.reshape(tf.convert_to_tensor(encoder_input), [1, len(encoder_input)])
		print("encoder in size:", np.shape(encoder_input))
		print("encoder in:", encoder_input)

		# pass increasingly large substring into call(), using the accumulated_output as input into the decoder
		for i in range(SIMPLE_WINDOW_SIZE):

			# pad decoder input so that it is [SIMPLE_WINDOW_SIZE,]
			decoder_input = accumulated_output[:SIMPLE_WINDOW_SIZE]
			decoder_input = [start_simple_id] + convert_to_id_single_string(simple_vocab, accumulated_output) + [stop_simple_id] + [pad_simple_id] * (SIMPLE_WINDOW_SIZE - len(decoder_input)-2)
			decoder_input = tf.reshape(tf.convert_to_tensor(decoder_input), [1, len(decoder_input)])


			# decoder_input = np.ones((1,220))
			
			# print("decoder in size:", np.shape(decoder_input))
			# print("decoder in:", decoder_input)

			probs = tf.squeeze(model.call([encoder_input, decoder_input])) #[(sub)_sentence_len x simple_vocab_size]
			# add newly predicted word to acc_output
			# print("probs of i: ", probs[i].shape)
			# print("probs of i max: ", tf.reduce_max(probs[i]))
			# print("probs of i: ", probs[i])
			new_id = tf.argmax(probs[i]) #TODO get argmax for curr word only

			# exit case
			if new_id == stop_simple_id or len(accumulated_output) == SIMPLE_WINDOW_SIZE-2:
				print("inference out: ", " ".join(accumulated_output))
				return " ".join(accumulated_output)

			position = vocab_idx_list.index(new_id)
			accumulated_output.append(vocab_word_list[position])

			if i % 20 == 0:
				print("iteration: ", i)
				print("new predicted word: ", vocab_word_list[position])


	def parse(text_input):
		"""
		Transforms a string into a list of words.

		:text_input: the text to be parsed
		:returns: parsed text as a list of words/punctuation marks
		"""
		words = re.split(r'\s|(?=[^A-Za-z0-9])|(?<=[\'|"|-|–|—])(?=[A-Za-z0-9])', text_input)
		while '' in words:
			words.remove('')
		return words
			
	def simplify(model, text_input, simplification_strength=1):
		"""
		Passes input to the trained model recursively by a given number of times to simplify the input by simplifcation_strength levels

		:param model: the trained transformer model
		:text_input: text to be simplified (string)
		:simplification_strength: number of times text_input should be simplified/passed into the model (int)
		:returns: the simplified text as a string
		"""
		if simplification_strength < 1:
			return text_input
		else:
			processed_text = convert_to_id_single_string(complex_vocab, parse(text_input))
			simplified = call_inference(model, processed_text)
			return simplify(model, simplified, simplification_strength - 1)

	return simplify(model, text_input)

