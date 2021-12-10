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

def get_model():
	cur_dir = os.path.dirname(os.path.abspath(__file__))

	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
	model = Simplifier_Transformer(*model_args)
	
	model((np.zeros((64, 420)), np.zeros((64, 220))))
	
	model.load_weights(cur_dir + "/model.h5")
	
	model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)

	return model

def get_test_dataset():
	print("Running preprocessing...")

	train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data(data_root+'/wiki_normal_train.txt', data_root+'/wiki_simple_train.txt', data_root+'/wiki_normal_test.txt', data_root+'/wiki_simple_test.txt')

	test_complex = test_complex[:1000, :]

	test_simple_trunc = test_simple[:1000, :-1]

	labels = test_simple[:1000, 1:]

	test_dataset = tf.data.Dataset.from_tensor_slices((test_complex, test_simple_trunc, labels))
	test_dataset = test_dataset.batch(64)
	return test_dataset

def evaluate_main(text_input, simplification_strength=1):
	model = get_model()
 
	test_dataset = get_test_dataset()
 
	score = model.evaluate(test_dataset, verbose=0)
	return score

