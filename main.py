import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from simplifier_model import Simplifier_Transformer
import sys
import random
import re


UNK_TOKEN = "*UNK*"
print("Running preprocessing...")
# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./wiki_normal_train.txt','./wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/wiki_normal_train.txt','./dummy_data/wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/fls.txt','./dummy_data/els.txt','./dummy_data/flt.txt','./dummy_data/elt.txt')
vocab_word_list = list(simple_vocab.keys())
vocab_idx_list = list(simple_vocab.values())
start_simple_id = simple_vocab["*START*"]
stop_simple_id = simple_vocab["*STOP*"]
print("Preprocessing complete.")

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
	# for i in range(0, num_batches*model.batch_size, model.batch_size):
	for i in range(0, 3*model.batch_size, model.batch_size):
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

def call_inference(model, input_ids):
	"""
	Used for inference without ground truth labels. Works by calling call() recursively,
	using the decoder output of the previous iteration each time (starting with the <START> token)
	as the new decoder input
	:param input_ids: 1D tensor of word ids of the input text string, [sentence_length,]
	:return: full string corresponding the the input_ids
	"""
	encoder_input = tf.reshape(input_ids, [1, tf.size(input_ids)]) #reshape to [1, sent_len]
	accumulated_output = [] # first word to be generated is after start token
	
	# pass increasingly large substring into call(), using the accumulated_output as input into the decoder
	for i in range(SIMPLE_WINDOW_SIZE):
		print("iteration: ", i)

		# pad decoder input so that it is [SIMPLE_WINDOW_SIZE,]
		decoder_input = []
		# for line in simple:
		decoder_input = accumulated_output[:SIMPLE_WINDOW_SIZE]
		print("decoder input L139: ", decoder_input)
		decoder_input = [start_simple_id] + convert_to_id_single_string(simple_vocab, accumulated_output) + [stop_simple_id] + [simple_padding_index] * (SIMPLE_WINDOW_SIZE - len(decoder_input)-2)
		
		decoder_input = tf.reshape(tf.convert_to_tensor(decoder_input), [1, len(decoder_input)])
		# decoder_input = np.ones((1,220))
		print("decoder in size:", np.shape(decoder_input))
		print("decoder in:", decoder_input)
		probs = tf.squeeze(model.call([encoder_input, decoder_input])) #[(sub)_sentence_len x simple_vocab_size]
		# add newly predicted word to acc_output
		new_id = tf.argmax(probs[i]) #TODO get argmax for curr word only
		if new_id == stop_simple_id or len(accumulated_output) == SIMPLE_WINDOW_SIZE-1:
			print("inference out: ", " ".join(accumulated_output))
			return " ".join(accumulated_output)

		position = vocab_idx_list.index(new_id)
		print("new predicted word: ", vocab_word_list[position])
		accumulated_output.append(vocab_word_list[position])

		



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
	processed_text = convert_to_id(model.complex_vocab, parse(text_input))
	# probs = model.call(processed_text)
	simplified = call_inference(model, processed_text)
	if simplification_strength == 1:
		return simplified
	else:
		return simplify(model, simplified, simplification_strength - 1)
	
def main():	
	

	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab))
	model = Simplifier_Transformer(*model_args) 
	
	# Train and Test Model for 1 epoch.
	print("==================TRAINING=================")
	train(model, train_complex, train_simple, simple_padding_index)
	print("===================TESTING=================")
	call_inference(model, np.ones((420)))
	# call_inference(model, parse("A stroke is a medical condition in which poor blood flow to the brain results in cell death . There are two main types of stroke : ischemic , due to lack of blood flow , and hemorrhagic , due to bleeding . Both result in parts of the brain not functioning properly . Signs and symptoms of a stroke may include an inability to move or feel on one side of the body , problems understanding or speaking , dizziness , or loss of vision to one side . Signs and symptoms often appear soon after the stroke has occurred . If symptoms last less than one or two hours it is known as a transient ischemic attack ( TIA ) or mini-stroke . A hemorrhagic stroke may also be associated with a severe headache . The symptoms of a stroke can be permanent ."))
	# call_inference(np.array("According to Indian law , no formality is needed during the procedure of arrest . The arrest can be made by a citizen , a police officer or a Magistrate . The police officer needs to inform the person being arrested the full particulars of the person' s offence and that they are entitled to be released on bail if the offence fits the criteria for being bailable .".split()))
	test(model, test_complex, test_simple, simple_padding_index)
	print("===================TESTING COMPLETE=================")

	pass

if __name__ == '__main__':
	main()
