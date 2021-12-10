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
train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data('./wiki_normal_train.txt','./wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data('./dummy_data/wiki_normal_train.txt','./dummy_data/wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/fls.txt','./dummy_data/els.txt','./dummy_data/flt.txt','./dummy_data/elt.txt')
vocab_word_list = list(simple_vocab.keys())
vocab_idx_list = list(simple_vocab.values())
stop_complex_id = complex_vocab["*STOP*"]
pad_complex_id = complex_vocab["*PAD*"]
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
	for i in range(0, num_batches*model.batch_size, model.batch_size):
	# for i in range(0, 10*model.batch_size, model.batch_size):
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
		if i//model.batch_size % 5 == 0:
			print("batch ", i//model.batch_size)

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
	
def main():	
	
	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab))
	model = Simplifier_Transformer(*model_args) 
	
	# Train and Test Model for 1 epoch.
	print("==================TRAINING=================")
	train(model, train_complex, train_simple, pad_simple_id)
	print("===================TESTING=================")
	call_inference(model, list(range(350)))
	call_inference(model, convert_to_id_single_string(complex_vocab, parse("A stroke is a medical condition in which poor blood flow to the brain results in cell death . There are two main types of stroke : ischemic , due to lack of blood flow , and hemorrhagic , due to bleeding . Both result in parts of the brain not functioning properly . Signs and symptoms of a stroke may include an inability to move or feel on one side of the body , problems understanding or speaking , dizziness , or loss of vision to one side . Signs and symptoms often appear soon after the stroke has occurred . If symptoms last less than one or two hours it is known as a transient ischemic attack ( TIA ) or mini-stroke . A hemorrhagic stroke may also be associated with a severe headache . The symptoms of a stroke can be permanent .")))
	call_inference(model, convert_to_id_single_string(complex_vocab, parse("Indian prime minister *UNK* *UNK* selected *UNK* as one of his nine *UNK* called *UNK* in 2014 for the *UNK* *UNK* *UNK* , a national *UNK* campaign by the Government of India . She *UNK* her support to the campaign by cleaning and *UNK* a *UNK* *UNK* in Mumbai , and urged people to maintain the *UNK* . In 2015 , she voiced People for the *UNK* Treatment of *UNK* ( *UNK* s ) *UNK* *UNK* elephant named *UNK* , who visited schools across the United States and Europe to *UNK* kids about elephants and captivity , and to *UNK* people to *UNK* *UNK* .")))
	string = "A stroke is a medical condition in which poor blood flow to the brain results in cell death . There are two main types of stroke : ischemic , due to lack of blood flow , and hemorrhagic , due to bleeding . Both result in parts of the brain not functioning properly . Signs and symptoms of a stroke may include an inability to move or feel on one side of the body , problems understanding or speaking , dizziness , or loss of vision to one side . Signs and symptoms often appear soon after the stroke has occurred . If symptoms last less than one or two hours it is known as a transient ischemic attack ( TIA ) or mini-stroke . A hemorrhagic stroke may also be associated with a severe headache . The symptoms of a stroke can be permanent ."
	print("=============SIMPLIFY================")
	simplify(model, string, 2)
	test(model, test_complex, test_simple, pad_simple_id)
	print("===================TESTING COMPLETE=================")

	pass

if __name__ == '__main__':
	main()