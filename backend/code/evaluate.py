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

import datetime

from tensorboard.plugins.hparams import api as hp

from metrics import custom_loss, AccWeightedSum, Perplexity

def evaluate_main(text_input, simplification_strength=1):
	print("Running preprocessing...")
 
	cur_dir = os.path.dirname(os.path.abspath(__file__))
	data_root = os.path.dirname(cur_dir) + '/data'

	train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, pad_simple_id = get_data(data_root+'/wiki_normal_train.txt', data_root+'/wiki_simple_train.txt', data_root+'/wiki_normal_test.txt', data_root+'/wiki_simple_test.txt')

	test_complex = test_complex[:1000, :]

	test_simple_trunc = test_simple[:1000, :-1]

	labels = test_simple[:1000, 1:]

	test_dataset = tf.data.Dataset.from_tensor_slices((test_complex, test_simple_trunc, labels))
	test_dataset = test_dataset.batch(64)

	# ====
	ADAM_LR = hp.HParam('adam_lr', hp.Discrete([0.001]))
	EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([50]))
	hparams = {
		'adam_lr': ADAM_LR.domain.values[0],
		'embedding_size': EMBEDDING_SIZE.domain.values[0],
	}

	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
	model = Simplifier_Transformer(*model_args)

	model((np.zeros((64, 420)), np.zeros((64, 220))))
	
	model.load_weights(cur_dir + "/model.h5")
	
	model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)
	
	# ======
 
	score = model.evaluate(test_dataset, verbose=0)
	return score

