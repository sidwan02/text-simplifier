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


"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
"""


physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
	tf.config.experimental.set_memory_growth(device, True)


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


	
def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"SAVE", "TUNE", "LOAD"}:
		print("USAGE: python main.py <Model Type>")
		print("<Model Type>: [SAVE/TUNE/LOAD]")
		exit()
  
	
		
	# UNK_TOKEN = "*UNK*"
	# print("Running preprocessing...")
	# train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data(data_root + '/wiki_normal_train.txt', data_root + '/wiki_simple_train.txt', data_root + '/wiki_normal_test.txt', data_root + '/wiki_simple_test.txt')
	# # train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/wiki_normal_train.txt','./dummy_data/wiki_simple_train.txt','./wiki_normal_test.txt','./wiki_simple_test.txt')
	# # train_simple, test_simple, train_complex, test_complex, simple_vocab, complex_vocab, simple_padding_index = get_data('./dummy_data/fls.txt','./dummy_data/els.txt','./dummy_data/flt.txt','./dummy_data/elt.txt')
	# print("Preprocessing complete.")

	cur_dir = os.path.dirname(os.path.abspath(__file__))
	data_root = os.path.dirname(cur_dir) + '/data'
	print(data_root)


	
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


	with open(cur_dir + '/simple_vocab.pkl', 'wb+') as f:
		pickle.dump(simple_vocab, f)

	with open(cur_dir + '/complex_vocab.pkl', 'wb+') as f:
		pickle.dump(complex_vocab, f)

	print("Preprocessing complete.")


	
	print("simple_padding_index ======================: ", pad_simple_id)
 
	# print("train_complex.shape ==========", train_complex.shape)	
 
	train_complex = train_complex[:, :]
 
	train_simple_trunc = train_simple[:, :-1]

	labels = train_simple[:, 1:]

	# ============
	# print("train_complex.shape: ", train_complex.shape)
	# print("train_simple_trunc.shape: ", train_simple_trunc.shape)
	# print("labels.shape: ", labels.shape)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_complex, train_simple_trunc, labels))
	train_dataset = train_dataset.batch(64)

	# model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])
	
	test_complex = test_complex[:, :]

	test_simple_trunc = test_simple[:, :-1]

	labels = test_simple[:, 1:]

	test_dataset = tf.data.Dataset.from_tensor_slices((test_complex, test_simple_trunc, labels))
	test_dataset = test_dataset.batch(64)

	# tf.data.experimental.save(train_dataset, cur_dir + "/train_dataset")
	tf.data.experimental.save(test_dataset, cur_dir + "/test_dataset")

	def save_trained_weights(hparams):
		model_args = (COMPLEX_WINDOW_SIZE, len(complex_vocab), SIMPLE_WINDOW_SIZE, len(simple_vocab), hparams)
		model = Simplifier_Transformer(*model_args) 
	 
		model.compile(optimizer=model.optimizer, loss=custom_loss, metrics=[AccWeightedSum(), Perplexity()], run_eagerly=True)
	 
		model.fit(
			train_dataset, 
			epochs=10, 
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
		
		# INFERENCE TESTING ==============
		# call_inference(model, list(range(350)))
		# call_inference(model, convert_to_id_single_string(complex_vocab, parse("A stroke is a medical condition in which poor blood flow to the brain results in cell death . There are two main types of stroke : ischemic , due to lack of blood flow , and hemorrhagic , due to bleeding . Both result in parts of the brain not functioning properly . Signs and symptoms of a stroke may include an inability to move or feel on one side of the body , problems understanding or speaking , dizziness , or loss of vision to one side . Signs and symptoms often appear soon after the stroke has occurred . If symptoms last less than one or two hours it is known as a transient ischemic attack ( TIA ) or mini-stroke . A hemorrhagic stroke may also be associated with a severe headache . The symptoms of a stroke can be permanent .")))
		# call_inference(model, convert_to_id_single_string(complex_vocab, parse("Indian prime minister *UNK* *UNK* selected *UNK* as one of his nine *UNK* called *UNK* in 2014 for the *UNK* *UNK* *UNK* , a national *UNK* campaign by the Government of India . She *UNK* her support to the campaign by cleaning and *UNK* a *UNK* *UNK* in Mumbai , and urged people to maintain the *UNK* . In 2015 , she voiced People for the *UNK* Treatment of *UNK* ( *UNK* s ) *UNK* *UNK* elephant named *UNK* , who visited schools across the United States and Europe to *UNK* kids about elephants and captivity , and to *UNK* people to *UNK* *UNK* .")))
		# string = "A stroke is a medical condition in which poor blood flow to the brain results in cell death . There are two main types of stroke : ischemic , due to lack of blood flow , and hemorrhagic , due to bleeding . Both result in parts of the brain not functioning properly . Signs and symptoms of a stroke may include an inability to move or feel on one side of the body , problems understanding or speaking , dizziness , or loss of vision to one side . Signs and symptoms often appear soon after the stroke has occurred . If symptoms last less than one or two hours it is known as a transient ischemic attack ( TIA ) or mini-stroke . A hemorrhagic stroke may also be associated with a severe headache . The symptoms of a stroke can be permanent ."
		# print("=============SIMPLIFY================")
		# simplify(model, string, 2)

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
			epochs=1, 
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
		
		ADAM_LR = hp.HParam('adam_lr', hp.Discrete([0.001, 0.005]))
		EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([40, 50]))
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
