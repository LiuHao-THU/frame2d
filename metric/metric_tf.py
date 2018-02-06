import tensorflow as tf 

"""
binary classification include:
1. binary cross entropy 
2. dice coef
3. mean square error
4. normalized mean square
5. abosulute different error
6. hard dice coef
milti_classification include 
1. soft_cross_entropy with logits
2. mean i
"""

def loss_dense(logits, labels):
    """	calculate loss using sparse_softmax_cross_entropy_with_logits input must be int32 or int64 no conside of batch
		logits:  [batch_size, img_raw, img_col,num_classes]
		labels: [batch_size,img_raw, img_col, 1]
    """
    input_shape = logits.get_shape().as_list()
    with tf.variable_scope('softmax_cross_entropy_with_logits'):
        #convert the labels to one one

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=logits, logits=labels, name=name))

def loss_sparse(logits, labels, axis = [0,1,2,3]):
    """logits:  [batch_size, img_raw, img_col,num_classes]
       labels: [batch_size,img_raw, img_col, 1]
    """
    input_shape = logits.get_shape().as_list()
    with tf.variable_scope('softmax_cross_entropy_with_logits'):
        pred_f = tf.reshape(logits, [input_shape[0], input_shape[-1]])
        label_f = tf.reshape(labels, [input_shape[0], -1])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_f, labels=label_f)
        loss = tf.reduce_mean(loss, name='loss')
        return loss

def accuracy(logits, labels, axis = [0,1,2,3]):
    """
    logits: [batch_size, img_raw, img_col,num_classes]
    labels: [batch_size,img_raw, img_col, 1]
    """
    with tf.name_scope("accuracy"):
        pred = tf.argmax(logits, 3)  #[batch_size,img_raw, img_col, 1]
        pred_f = tf.reshape(pred, [-1])
        label_f = tf.reshape(labels, [-1])
        correct_prediction = tf.equal(pred_f, label_f)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy

def mean_iou(logits, labels, axis = [0,1,2,3]):
    """
    logits: [batch_size, img_raw, img_col,num_classes]
    labels: [batch_size,img_raw, img_col, 1]
    return
    mean_iou: A Tensor representing the mean intersection-over-union.
    update_op: An operation that increments the confusion matrix.
    """
    input_shape = logits.get_shape().as_list()
    with tf.name_scope("mean_iou"):
        img_pred = tf.argmax(logits, 3)
        iou, iou_op = tf.metrics.mean_iou(labels, img_pred, input_shape[-1])
        return iou, iou_op


def mean_squared_error(logits, labels, axis = [0,1,2,3]):
    """
    logits : [batch_size, w, h, num_classes]  float32 
    labels : [batch_size, w, h, 1]            int32
    """
    with tf.name_scope("mean_squared_error_loss"):#[batch_size, w, h, c]
        mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(logits, labels), [1, 2, 3]))
        return mse


def normalized_mean_square_error(logits, labels, axis = [0,1,2,3]):
    """
    logits : [batch_size, w, h, num_classes]
    labels : [batch_size, w, h, 1]
    """
    with tf.name_scope("normalized_mean_square_error"):
        nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(logits, labels), axis=[1,2,3]))
        nmse_b = tf.sqrt(tf.reduce_sum(tf.square(labels), axis=[1,2,3]))
        nmse = tf.reduce_mean(nmse_a / nmse_b)
    return nmse

def absolute_difference_error(logits, labels, axis = [0,1,2,3]):
    """
    logits : [batch_size, w, h, num_classes]
    labels : [batch_size, w, h, 1]
    """
    with tf.name_scope("absolute_difference_error"):
        loss = tf.reduce_mean(tf.reduce_mean(tf.abs(logits - labels), [1, 2, 3]))
        return loss




def dice_coef(logits, labels, loss_type='sorensen', smooth=1e-5, axis=[1,2,3]):
    """ Soft dice (Sørensen or Jaccard) for binary classification problem 
        logits : [batch_size, w, h, 1]
        labels : [batch_size, w, h, 1]
    """
    # logits = tf.clip_by_value(logits,1e-10,1.0)   
    with tf.name_scope("dice_coef"):     
        inse = tf.reduce_sum(logits * labels, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(logits * logits, axis = axis)
            r = tf.reduce_sum(labels * labels, axis = axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(logits, axis = axis)
            r = tf.reduce_sum(labels, axis = axis)

        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice


def dice_hard_coef(logits, labels, axis = [0,1,2,3], threshold=0.5, smooth=1e-5):
    """ for binary classification problem 
        logits : [batch_size, w, h, num_classes]
        labels : [batch_size, w, h, 1]
    """
    with tf.name_scope("dice_hard_coef"): 
	    output = tf.cast(logits > threshold, dtype=tf.float32)
	    target = tf.cast(labels > threshold, dtype=tf.float32)
	    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
	    l = tf.reduce_sum(output, axis=axis)
	    r = tf.reduce_sum(target, axis=axis)
	    hard_dice = (2. * inse + smooth) / (l + r + smooth)
	    hard_dice = tf.reduce_mean(hard_dice)
	    return hard_dice


