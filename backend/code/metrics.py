import tensorflow as tf

def create_mask(labels, eng_padding_index=10):
    mask = tf.where(labels == eng_padding_index, 0, 1)
    batch_valid_tokens = tf.cast(tf.math.reduce_sum(mask), dtype=tf.float32)
    return mask, batch_valid_tokens
        

def custom_loss(y_true, y_pred):
    probs, labels = y_pred, y_true
    mask, _ = create_mask(labels)

    def loss(probs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param probs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        # assert probs.shape == (
        # self.batch_size, self.english_window_size, self.english_vocab_size)
        # assert labels.shape == (self.batch_size, self.english_window_size)
        # assert mask.shape == (self.batch_size, self.english_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, probs), mask))

    return loss(probs, labels, mask)

class AccWeightedSum(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, **kwargs):
        super(AccWeightedSum,self).__init__(name='acc_weighted_sum_metric',**kwargs) # handles base args (e.g., dtype)
        self.total_tokens = self.add_weight("total_tokens", initializer="zeros")
        self.total_weighted_acc = self.add_weight("total_weighted_acc", initializer="zeros")
        
    def reset_state(self):
        self.total_tokens.assign(0)
        self.total_weighted_acc.assign(0)
            
    def update_state(self, y_true, y_pred, sample_weight=None): # y_true = labels; y_pred = probs
        probs, labels = y_pred, y_true
        # print("probs: ", probs)

        mask, batch_valid_tokens = create_mask(labels)
        acc = self.accuracy_function(probs, labels, mask)

        # print("acc: ", acc)
        # print("batch_valid_tokens: ", batch_valid_tokens)

        self.total_tokens.assign_add(batch_valid_tokens)
        self.total_weighted_acc.assign_add(acc * batch_valid_tokens)

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        labels = tf.cast(labels, dtype=tf.int64)

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(
            tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
        return accuracy

    def result(self):
        # print("total_weighted_acc: ", self.total_weighted_acc)
        # print("total_tokens: ", self.total_tokens)
        ans = self.total_weighted_acc / self.total_tokens
        # print("ans: ", ans)
        return ans
        
class Perplexity(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, **kwargs):
        super(Perplexity,self).__init__(name='perplexity_metric',**kwargs) # handles base args (e.g., dtype)
        self.acc_loss = self.add_weight("acc_loss", initializer="zeros")
        self.total_valid_tokens = self.add_weight("total_valid_tokens", initializer="zeros")
        
    def reset_state(self):
        self.acc_loss.assign(0)
        self.total_valid_tokens.assign(0)
            
    def update_state(self, y_true, y_pred, sample_weight=None): # y_true = labels; y_pred = probs
        probs, labels = y_pred, y_true
        _, batch_valid_tokens = create_mask(labels)
        loss = custom_loss(y_pred=probs, y_true=labels)

        self.acc_loss.assign_add(loss)
        self.total_valid_tokens.assign_add(batch_valid_tokens)
        
    def result(self):
        ans = tf.exp(self.acc_loss / self.total_valid_tokens)
        # print("ans: ", ans)
        return ans
