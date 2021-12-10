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

"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
"""

"""
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
	tf.config.experimental.set_memory_growth(device, True)
"""

import datetime

from tensorboard.plugins.hparams import api as hp

from metrics import custom_loss, AccWeightedSum, Perplexity

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

hparams_log_dir = 'logs/hparam_tuning/' + current_time

l = np.arange(0.0001, 0.01 + 0.0001, 0.0005)
print("l: ", l)

ADAM_LR = hp.HParam('adam_lr', hp.Discrete(l.tolist()))

hp.hparams_config(
	hparams=[ADAM_LR],
	# https://stackoverflow.com/questions/56852300/hyperparameter-tuning-using-tensorboard-plugins-hparams-api-with-custom-loss-fun
	metrics=[hp.Metric('AccWeightedSum', display_name='acc_weighted_sum'), hp.Metric('Perplexity', display_name='perplexity')],
	)


def train(model, train_complex, train_simple, simple_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_complex: complex train data (all data for training) of shape (num_sentences, COMPLEX_WINDOW_SIZE)
	:param train_simple: simple train data (all data for training) of shape (num_sentences, SIMPLE_WINDOW_SIZE+1)
	:param simple_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the complex sentences to be used by the encoder, 
	# and simple sentences to be used by the decoder
	# - The simple sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 

	# split decoder train data into inputs & outputs (forced learning)
	decoder_input = train_simple[:, :-1] #remove last sentence element
	decoder_labels = train_simple[:, 1:] #remove first sentence element
	# assert np.shape(decoder_input)[1] == 14
	# assert np.shape(decoder_labels)[1] == 14
	# print(np.shape(decoder_input)[1])
	# print(np.shape(decoder_labels)[1])

	# mask losses corrsponding with padding tokens => 0 (prevent training)
	mask = tf.cast(tf.not_equal(np.array(decoder_labels), simple_padding_index), dtype=tf.float32)

	# batch and train
	num_batches = np.shape(train_complex)[0] // model.batch_size
	print("num batches: ", num_batches)
	optimizer = model.optimizer
	for i in range(0, num_batches*model.batch_size, model.batch_size):
	# for i in range(0, 50*model.batch_size, model.batch_size):
		# batch data
		batch_encoder_input = train_complex[i:i+model.batch_size]
		batch_decoder_input = decoder_input[i:i+model.batch_size]
		batch_decoder_labels = decoder_labels[i:i+model.batch_size]
		batch_mask = mask[i:i+model.batch_size]
		# assert batch_mask.shape == np.shape(batch_decoder_labels)
		# forward pass
		with tf.GradientTape() as tape:
			probs = model.call([batch_encoder_input, batch_decoder_input])
			loss = model.loss_function(probs, batch_decoder_labels, batch_mask)
		if i//model.batch_size % 5 == 0:
			print("batch ", i//model.batch_size)
			print("loss: ", loss)		
		# backprop
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

def test(model, test_complex, test_simple, simple_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_complex: complex test data (all data for testing) of shape (num_sentences, COMPLEX_WINDOW_SIZE)
	:param test_simple: simple test data (all data for testing) of shape (num_sentences, SIMPLE_WINDOW_SIZE+1)
	:param simple_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""
	# Note: Follow the same procedure as in train() to construct batches of data!
	decoder_input = test_simple[:, :-1] #remove last sentence element
	decoder_labels = test_simple[:, 1:] #remove first sentence element

	# mask losses corrsponding with padding tokens => 0 (prevent training)
	mask = tf.cast(tf.not_equal(np.array(decoder_labels), simple_padding_index), dtype=tf.float32)
	

	# batch and train
	num_batches = np.shape(test_complex)[0] // model.batch_size
	print("num batches: ", num_batches)

	losses = 0 # accumulate per-sentence loss for perplexity
	acc = 0 # accumulate per-symbol loss
	num_non_padding_tokens = 0
	for i in range(0, num_batches*model.batch_size, model.batch_size):
	# for i in range(0, 50*model.batch_size, model.batch_size):
		# batch data
		batch_encoder_input = test_complex[i:i+model.batch_size]
		batch_decoder_input = decoder_input[i:i+model.batch_size]
		batch_decoder_labels = decoder_labels[i:i+model.batch_size]
		batch_mask = mask[i : i + model.batch_size]
		batch_num_non_padding_tokens = tf.reduce_sum(batch_mask)
		num_non_padding_tokens += batch_num_non_padding_tokens
		# accumulate losses and accs
		probs = model.call([batch_encoder_input, batch_decoder_input])
		losses += model.loss_function(probs, batch_decoder_labels, batch_mask) #loss func returns sum of losses in batch
		acc += model.accuracy_function(probs, batch_decoder_labels, batch_mask) * batch_num_non_padding_tokens	

	perplexity = tf.exp( losses / num_non_padding_tokens ) 
	avg_acc = acc / num_non_padding_tokens
	print("PERPLEXITY: ", perplexity)
	print("ACC: ", avg_acc)
	return perplexity, avg_acc


#### NOTE: commented out because lambda function gives syntax error

# def probs_to_words(probs):
# 	"""
# 	helper function for converting decoder output (after being passed through softmax) to a sequence of words for use in simplify()

# 	:param probs: The word probabilities as a tensor, [window_size x vocab_size]
# 	:returns: words corresponding to the probabilities in a sentence

# 	"""
# 	probable_tokens = tf.argmax(probs, axis=1)
	# probable_words = tf.map_fn(lambda token: simple_vocab[token], probable_tokens)
	# probable_sentence = tf.join(probable_words, separator=' ')
	# probable_sentence = tf.strings.as_string(probable_sentence)

	# return re.sub('\s(?=[^A-Za-z0-9])', '', probable_sentence)

def parse(text_input):
	"""
 	Transforms a string into a list of sentences, which are lists of words.
  
	:text_input: the text to be parsed
	:returns: parsed text
	"""
	non_alpha_numeric = '[^A-Za-z0-9]'
	processed_text = []
	sentences = re.split('(?<=\.\s)', text_input)
	while '' in sentences:
		sentences.remove('')
  
	for sentence in sentences:
		split_sentence = re.split(r'\s|(?=[^A-Za-z0-9])|(?<=[\'|"|-|–|—])(?=[A-Za-z0-9])', sentence)
		while '' in split_sentence:
			split_sentence.remove('')
		processed_text(split_sentence)
	return processed_text
		
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
	processed_text = convert_to_id(model.complex_vocab, parse(text_input))
	probs = model.call(processed_text)
	simplified = probs_to_words(text_input)
	if simplification_strength == 1:
		return simplified
	else:
		return simplify(model, simplified, simplification_strength - 1)
	
def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"SAVE", "TUNE", "LOAD"}:
		print("USAGE: python main.py <Model Type>")
		print("<Model Type>: [SAVE/TUNE/LOAD]")
		exit()
  
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	data_root = os.path.dirname(cur_dir) + '/data'
	print(data_root)
		
	UNK_TOKEN = "*UNK*"
	print("Running preprocessing...")
	train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data(data_root + '/wiki_normal_train.txt', data_root + '/wiki_simple_train.txt', data_root + '/wiki_normal_test.txt', data_root + '/wiki_simple_test.txt')
	# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/wiki_normal_train.txt','./dummy_data/wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
	# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/fls.txt','./dummy_data/els.txt','./dummy_data/flt.txt','./dummy_data/elt.txt')
	print("Preprocessing complete.")
 
	print("simple_padding_index ======================: ", simple_padding_index)
 
	# print("train_complex.shape ==========", train_complex.shape)	
 
	train_complex = train_complex[:1000, :]
 
	train_simple_trunc = train_simple[:1000, :-1]

	labels = train_simple[:1000, 1:]

	# ============
	# print("train_complex.shape: ", train_complex.shape)
	# print("train_simple_trunc.shape: ", train_simple_trunc.shape)
	# print("labels.shape: ", labels.shape)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_complex, train_simple_trunc, labels))
	train_dataset = train_dataset.batch(64)

	# model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])
	
	test_complex = test_complex[:1000, :]

	test_simple_trunc = test_simple[:1000, :-1]

	labels = test_simple[:1000, 1:]

	test_dataset = tf.data.Dataset.from_tensor_slices((test_complex, test_simple_trunc, labels))
	test_dataset = test_dataset.batch(64)


	def save_trained_weights(hparams):
		model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
		model = Simplifier_Transformer(*model_args) 
	 
		model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)
	 
		model.fit(
			train_dataset, 
			epochs=3, 
			# callbacks=[
			#     keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, update_freq='batch', embeddings_freq=1), 
			#     hp.KerasCallback(logdir, hparams)
			#     ], 
			validation_data=test_dataset
			)
	 
		model((np.zeros((64, 420)), np.zeros((64, 220))))
		
		model.save_weights(cur_dir + "/model.h5")

	def evaluate_model_from_loaded_weights(hparams):
		model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
		model = Simplifier_Transformer(*model_args)
		
		model((np.zeros((64, 420)), np.zeros((64, 220))))
		
		model.load_weights(cur_dir + "/model.h5")
		
		model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)
		
		score = model.evaluate(test_dataset, verbose=0)

		print("score: ", score)

		return score
	
	def run(hparams, logdir):
		print("hparams: ", hparams)

		# model_args = (FRENCH_WINDOW_SIZE, len(french_vocab),
		#             ENGLISH_WINDOW_SIZE, len(english_vocab), hparams)
		# if sys.argv[1] == "RNN":
		#     model = RNN_Seq2Seq(*model_args)
		# elif sys.argv[1] == "TRANSFORMER":
		# model = Transformer_Seq2Seq(*model_args)
		model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
		model = Simplifier_Transformer(*model_args)

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
			validation_data=test_dataset
			)

	if sys.argv[1] == "SAVE":
		ADAM_LR = hp.HParam('adam_lr', hp.Discrete([0.001]))
		EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([50]))
		hparams = {
			'adam_lr': ADAM_LR.domain.values[0],
			'embedding_size': EMBEDDING_SIZE.domain.values[0],
		}

		save_trained_weights(hparams)

	elif sys.argv[1] == "LOAD":
		ADAM_LR = hp.HParam('adam_lr', hp.Discrete([0.001]))
		EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([50]))
		hparams = {
			'adam_lr': ADAM_LR.domain.values[0],
			'embedding_size': EMBEDDING_SIZE.domain.values[0],
		}

		evaluate_model_from_loaded_weights(hparams)

	elif sys.argv[1] == "TUNE":
		# l = np.arange(0.0001, 0.01 + 0.0001, 0.0005)
		# print("l: ", l)
		# ADAM_LR = hp.HParam('adam_lr', hp.Discrete(l.tolist()))
		
		ADAM_LR = hp.HParam('adam_lr', hp.Discrete([0.015, 0.001, 0.005]))
		EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([40, 50, 60]))
		# hparams = {
		# 	'adam_lr': ADAM_LR.domain.values[0],
   		# 	'embedding_size': EMBEDDING_SIZE.domain.values[0],
		# }
		
		# av.setup_visualization(enable=True)
		session_num = 0

		# https://stackoverflow.com/questions/56559627/what-are-hp-discrete-and-hp-realinterval-can-i-include-more-values-in-hp-realin
		for adam_lr in ADAM_LR.domain.values:
			for embedding_size in EMBEDDING_SIZE.domain.values:
				hparams = {
					'adam_lr': adam_lr,
					'embedding_size': embedding_size,
				}
				run_name = "run-%d" % session_num
				print('--- Starting trial: %s' % run_name)
				print({h: hparams[h] for h in hparams})
				run(hparams, hparams_log_dir + run_name)
				session_num += 1
	
	
if __name__ == '__main__':
	main()