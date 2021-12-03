class AccWeightedSum(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, **kwargs):
        super(LossPerSymbol,self).__init__(name='acc_weighted_sum_metric',**kwargs) # handles base args (e.g., dtype)
        self.total_tokens = self.add_weight("total_tokens", initializer="zeros")
        self.total_weighted_acc = self.add_weight("total_weighted_acc", initializer="zeros")
        
    def reset_states(self):
        self.total_tokens.assign(0)
        self.total_weighted_acc.assign(0)
            
    def update_state(self, y_true, y_pred, sample_weight=None): # y_true = labels; y_pred = probs
        probs, labels = y_pred, y_true
        mask, batch_valid_tokens = self.create_mask(labels)
        acc = self.accuracy_function(probs, labels, mask)

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

    def create_mask(self, labels, eng_padding_index=0):
        mask = tf.where(labels == eng_padding_index, 0, 1)
        batch_valid_tokens = tf.cast(tf.math.reduce_sum(mask), dtype=tf.float32)
        return mask, batch_valid_tokens
        
    def result(self):
        return self.total_weighted_acc / self.total_tokens
    
class Perplexity(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, **kwargs):
        super(LossPerSymbol,self).__init__(name='perplexity_metric',**kwargs) # handles base args (e.g., dtype)
        self.acc_loss = self.add_weight("acc_loss", initializer="zeros")
        self.total_valid_tokens = self.add_weight("total_valid_tokens", initializer="zeros")
        
    def reset_states(self):
        self.acc_loss.assign(0)
        self.total_valid_tokens.assign(0)
            
    def update_state(self, y_true, y_pred, sample_weight=None): # y_true = labels; y_pred = probs
        probs, labels = y_pred, y_true
        mask, batch_valid_tokens = self.create_mask(labels)
        loss = self.loss_function(probs, labels, mask)

        self.acc_loss.assign_add(loss)
        self.total_valid_tokens.assign_add(batch_valid_tokens)


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        # assert prbs.shape == (
        # self.batch_size, self.english_window_size, self.english_vocab_size)
        # assert labels.shape == (self.batch_size, self.english_window_size)
        # assert mask.shape == (self.batch_size, self.english_window_size)

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs), mask))
    

    
    def create_mask(self, labels, eng_padding_index=0):
        mask = tf.where(labels == eng_padding_index, 0, 1)
        batch_valid_tokens = tf.cast(tf.math.reduce_sum(mask), dtype=tf.float32)
        return mask, batch_valid_tokens
        
    def result(self):
        return tf.exp(self.acc_loss / self.total_valid_tokens)

class LossPerSymbol(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, **kwargs):
        super(LossPerSymbol,self).__init__(name='loss_per_symbol_metric',**kwargs) # handles base args (e.g., dtype)
        self.total_loss = self.add_weight("total_loss", initializer="zeros")
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        self.total_loss.assign(0)
            
    def update_state(self, y_true, y_pred, sample_weight=None): # y_true = labels; y_pred = probs
        mask, batch_valid_tokens = self.create_mask(y_pred)
        self.total_loss.assign_add(loss / batch_valid_tokens)
        return self.total_cm

    def create_mask(self, labels, eng_padding_index=0):
        mask = tf.where(labels == eng_padding_index, 0, 1)
        batch_valid_tokens = tf.cast(tf.math.reduce_sum(mask), dtype=tf.float32)
        return mask, batch_valid_tokens
        
    def result(self):
        return self.process_confusion_matrix()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return precision,recall,f1
    
    def fill_output(self,output):
        results=self.result()
        for i in range(self.num_classes):
            output['precision_{}'.format(i)]=results[0][i]
            output['recall_{}'.format(i)]=results[1][i]
            output['F1_{}'.format(i)]=results[2][i]
    


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric,self).__init__(name='confusion_matrix_metric',**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
        
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        return self.process_confusion_matrix()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return precision,recall,f1
    
    def fill_output(self,output):
        results=self.result()
        for i in range(self.num_classes):
            output['precision_{}'.format(i)]=results[0][i]
            output['recall_{}'.format(i)]=results[1][i]
            output['F1_{}'.format(i)]=results[2][i]
    
