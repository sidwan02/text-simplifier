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
import pickle
import datetime

from tensorboard.plugins.hparams import api as hp

from metrics import custom_loss, AccWeightedSum, Perplexity

def evaluate_main():
	print("Running preprocessing...")
 
	cur_dir = os.path.dirname(os.path.abspath(__file__))

	test_dataset = tf.data.experimental.load(cur_dir + "/test_dataset")

	with open(cur_dir + '/simple_vocab.pkl', 'rb') as f:
		simple_vocab = pickle.load(f)

	with open(cur_dir + '/complex_vocab.pkl', 'rb') as f:
		complex_vocab = pickle.load(f)

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

