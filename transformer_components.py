import numpy as np
import tensorflow as tf
import numpy as np


def Attention_Matrix(K, Q, use_mask=False):
	"""
	This functions runs a single attention head.

	:param K: is [batch_size x window_size_keys x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention matrix
	"""
	
	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	# Check lecture slides for how to compute self-attention
	# Remember: 
	# - Q is [batch_size x window_size_queries x embedding_size]
	# - K is [batch_size x window_size_keys x embedding_size]
	# - Mask is [batch_size x window_size_queries x window_size_keys]


	# Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector. 
	# This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
	# Those weights are then used to create linear combinations of the corresponding values for each query.
	# Those queries will become the new embeddings.
	
	# 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)

	#transpose k:
	K_t = tf.transpose(K, [0, 2, 1]) #transpose inner dimensions (outer dimension=batch)
	# attention has shape [batch_sz, window_size_key]
	attention = tf.matmul(Q, K_t) / tf.math.sqrt(tf.cast(window_size_keys, tf.float32)) # divide by sqrt(key vector dim), must cast to float first
	if use_mask:
		# apply attention mask
		attention = attention + atten_mask
	attention = tf.nn.softmax(attention)
	return attention # shape [batch_size x window_size_queries x window_size_keys]




# class Atten_Head(tf.keras.layers.Layer):
# 	def __init__(self, input_size, output_size, use_mask):		
# 		super(Atten_Head, self).__init__()

# 		self.use_mask = use_mask

# 		# TODO:
# 		# Initialize the weight matrices for K, V, and Q.
# 		# They should be able to multiply an input_size vector to produce an output_size vector 
# 		# Hint: use self.add_weight(...)
# 		self.K = self.add_weight(shape=[input_size, output_size], initializer="random_normal", trainable=True)
# 		self.Q = self.add_weight(shape=[input_size, output_size], initializer="random_normal", trainable=True)
# 		self.V = self.add_weight(shape=[input_size, output_size], initializer="random_normal", trainable=True)
		
# 	@tf.function
# 	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
# 		"""
# 		This functions runs a single attention head.

# 		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
# 		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
# 		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
# 		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
# 		"""

# 		# - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this. 
# 		K = tf.tensordot(inputs_for_keys, self.K, [ [2], [0] ]) # [batch_size x WINDOW_SIZE x input_size ], [ input x output ] ==> [batch_size x WINDOW_SIZE x output_size ]
# 		Q = tf.tensordot(inputs_for_queries, self.Q, [ [2], [0] ])
# 		V = tf.tensordot(inputs_for_values, self.V, [ [2], [0] ])

# 		# - Call Attention_Matrix with the keys and queries, and with self.use_mask.
# 		# - Apply the attention matrix to the values to get a single attention head (collection of self-attention output vectors z_i)
# 		head = Attention_Matrix(K, Q, self.use_mask) 
# 		print("att head shape:", tf.matmul(head, V).shape)
# 		return tf.matmul(head, V)  # [batch_size x window_size x window_size] x [batch_size x window_size x output_size ] = [batch_size, window_size, output]



class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()
		sizes = []
		if emb_sz % 3 == 0:
			sizes = [emb_sz//3, emb_sz//3, emb_sz//3]
		elif emb_sz % 3 == 1:
			sizes = [math.floor(emb_sz/3), math.floor(emb_sz/3), math.ceil(emb_sz/3)]
		else:
			sizes = [math.floor(emb_sz/3), math.ceil(emb_sz/3), math.ceil(emb_sz/3)]

		self.use_mask = use_mask

		self.key_weight1 = self.add_weight(shape=(emb_sz, sizes[0]))
		self.value_weight1 = self.add_weight(shape=(emb_sz, sizes[0]))
		self.query_weight1 = self.add_weight(shape=(emb_sz, sizes[0]))

		self.key_weight2 = self.add_weight(shape=(emb_sz, sizes[1]))
		self.value_weight2 = self.add_weight(shape=(emb_sz, sizes[1]))
		self.query_weight2 = self.add_weight(shape=(emb_sz, sizes[1]))

		self.key_weight3 = self.add_weight(shape=(emb_sz, sizes[2]))
		self.value_weight3 = self.add_weight(shape=(emb_sz, sizes[2]))
		self.query_weight3 = self.add_weight(shape=(emb_sz, sizes[2]))

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		This functions runs a multiheaded attention layer.

		Requirements:
			- Splits data for 3 different heads of size embed_sz/3
			- Create three different attention heads
			- Concatenate the outputs of these heads together
			- Apply a linear layer

		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
		K1 = tf.tensordot(inputs_for_keys, self.key_weight1, [[2], [0]])
		V1 = tf.tensordot(inputs_for_values, self.value_weight1, [[2], [0]])
		Q1 = tf.tensordot(inputs_for_queries, self.query_weight1, [[2], [0]])

		K2 = tf.tensordot(inputs_for_keys, self.key_weight2, [[2], [0]])
		V2 = tf.tensordot(inputs_for_values, self.value_weight2, [[2], [0]])
		Q2 = tf.tensordot(inputs_for_queries, self.query_weight2, [[2], [0]])

		K3 = tf.tensordot(inputs_for_keys, self.key_weight3, [[2], [0]])
		V3 = tf.tensordot(inputs_for_values, self.value_weight3, [[2], [0]])
		Q3 = tf.tensordot(inputs_for_queries, self.query_weight3, [[2], [0]])

		attention_mat1 = Attention_Matrix(K1, Q1, self.use_mask)
		attention_mat2 = Attention_Matrix(K2, Q2, self.use_mask)
		attention_mat3 = Attention_Matrix(K3, Q3, self.use_mask)

		res1 = tf.matmul(attention_mat1, V1)
		res2 = tf.matmul(attention_mat2, V2)
		res3 = tf.matmul(attention_mat3, V3)

		return tf.concat([res1, res2, res3], 2)



class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):
		"""
		This functions creates a feed forward network as described in 3.3
		https://arxiv.org/pdf/1706.03762.pdf

		Requirements:
		- Two linear layers with relu between them

		:param inputs: input tensor [batch_size x window_size x embedding_size]
		:return: tensor [batch_size x window_size x embedding_size]
		"""
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder):
		super(Transformer_Block, self).__init__()

		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Multi_Headed(emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Multi_Headed(emb_sz,use_mask=False)

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):
		"""
		This functions calls a transformer block.

		There are two possibilities for when this function is called.
		    - if self.is_decoder == False, then:
		        1) compute unmasked attention on the inputs
		        2) residual connection and layer normalization
		        3) feed forward layer
		        4) residual connection and layer normalization

		    - if self.is_decoder == True, then:
		        1) compute MASKED attention on the inputs
		        2) residual connection and layer normalization
		        3) computed UNMASKED attention using context
		        4) residual connection and layer normalization
		        5) feed forward layer
		        6) residual layer and layer normalization

		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)

		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""

		atten_out = self.self_atten(inputs,inputs,inputs)
		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		if self.is_decoder:
			assert context is not None,"Decoder blocks require context"
			context_atten_out = self.self_context_atten(context,context,atten_normalized)
			context_atten_out+=atten_normalized
			atten_normalized = self.layer_norm(context_atten_out)

		ff_out=self.ff_layer(atten_normalized)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)




# === No need for positional encoding? ===

# class Position_Encoding_Layer(tf.keras.layers.Layer):
	# def __init__(self, window_sz, emb_sz):
	# 	super(Position_Encoding_Layer, self).__init__()
	# 	self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	# @tf.function
	# def call(self, x):
	# 	"""
	# 	Adds positional embeddings to word embeddings.    

	# 	:param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
	# 	:return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
	# 	"""
	# 	return x+self.positional_embeddings
