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
	model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
	model = Simplifier_Transformer(*model_args)
	
	model((np.zeros((64, 420)), np.zeros((64, 220))))
	
	model.load_weights(cur_dir + "/model.h5")
	
	model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)
 
	return model

def evaluate_main(text_input, simplification_strength=1):
	model = get_model()
	score = model.evaluate(test_dataset, verbose=0)
	return score

