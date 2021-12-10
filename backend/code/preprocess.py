import numpy as np
import tensorflow as tf
import numpy as np


##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
COMPLEX_WINDOW_SIZE = 420
SIMPLE_WINDOW_SIZE = 220
##########DO NOT CHANGE#####################

def pad_corpus(complex, simple):
	"""
	arguments are lists of COMPLEX, SIMPLE sentences. Returns [COMPLEX-sents, SIMPLE-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param complex: list of French sentences
	:param simple: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	COMPLEX_padded_sentences = []
	for line in complex:
		padded_COMPLEX = line[:COMPLEX_WINDOW_SIZE]
		padded_COMPLEX += [STOP_TOKEN] + [PAD_TOKEN] * (COMPLEX_WINDOW_SIZE - len(padded_COMPLEX)-1)
		COMPLEX_padded_sentences.append(padded_COMPLEX)

	SIMPLE_padded_sentences = []
	for line in simple:
		padded_SIMPLE = line[:SIMPLE_WINDOW_SIZE]
		padded_SIMPLE = [START_TOKEN] + padded_SIMPLE + [STOP_TOKEN] + [PAD_TOKEN] * (SIMPLE_WINDOW_SIZE - len(padded_SIMPLE)-1)
		SIMPLE_padded_sentences.append(padded_SIMPLE)

	print("complex shape:", np.shape(COMPLEX_padded_sentences))
	print("simple shape:", np.shape(SIMPLE_padded_sentences))
	return COMPLEX_padded_sentences, SIMPLE_padded_sentences

def build_vocab(sentences):
	"""
  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
  Convert sentences to indexed 

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """

	# for sentence in sentences:
		# print(np.shape(sentence))
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

def get_data(complex_training_file, simple_training_file, complex_test_file, simple_test_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param complex_training_file: Path to the complex training file.
	:param simple_training_file: Path to the simple training file.
	:param complex_test_file: Path to the complex test file.
	:param simple_test_file: Path to the simple test file.
	
	:return: Tuple of train containing:
	(2-d list or array with simple training sentences in vectorized/id form [num_sentences x (SIMPLE_WINDOW_SIZE + 1)] ),
	(2-d list or array with simple test sentences in vectorized/id form [num_sentences x (SIMPLE_WINDOW_SIZE + 1)]),
	(2-d list or array with complex training sentences in vectorized/id form [num_sentences x COMPLEX_WINDOW_SIZE]),
	(2-d list or array with complex test sentences in vectorized/id form [num_sentences x COMPLEX_WINDOW_SIZE]),
	simple vocab (Dict containg word->index mapping),
	complex vocab (Dict containg word->index mapping),
	simple padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index

	#1) Read English and French Data for training and testing (see read_data)
	complex_train = read_data(complex_training_file)
	complex_test = read_data(complex_test_file)
	simple_train = read_data(simple_training_file)
	simple_test = read_data(simple_test_file)
	#2) Pad training data (see pad_corpus)
	print("padding")
	padded_complex_train, padded_simple_train = pad_corpus(complex_train, simple_train)
	#3) Pad testing data (see pad_corpus)
	padded_complex_test, padded_simple_test = pad_corpus(complex_test, simple_test)
	#4) Build vocab for complex (see build_vocab)
	print("building vocab")
	complex_vocab, complex_pad_idx = build_vocab(padded_complex_train)
	#5) Build vocab for simple (see build_vocab)
	simple_vocab, simple_pad_idx = build_vocab(padded_simple_train)
	#6) Convert training and testing simple sentences to list of IDS (see convert_to_id)
	print("converting to ids")
	simple_train_ids = convert_to_id(simple_vocab, padded_simple_train)
	simple_test_ids = convert_to_id(simple_vocab, padded_simple_test)
	#7) Convert training and testing complex sentences to list of IDS (see convert_to_id)
	complex_train_ids = convert_to_id(complex_vocab, padded_complex_train)
	complex_test_ids = convert_to_id(complex_vocab, padded_complex_test)
	print(np.shape(simple_train_ids))
	# assert np.shape(simple_train_ids) == (351846, 15)
	print(np.shape(simple_test_ids))
	print(np.shape(complex_train_ids))
	print(np.shape(complex_test_ids))
	print("simple vocab: ", len(simple_vocab))
	return simple_train_ids, simple_test_ids, complex_train_ids, complex_test_ids, simple_vocab, complex_vocab, simple_pad_idx
	