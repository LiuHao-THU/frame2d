"""
A Trainable ResNet Class is defined in this file
Author: Kaihua Tang
"""
import math
import numpy as np
import tensorflow as tf
from functools import reduce
from configs import configs
class ResNet:
	# some properties
    """
    Initialize function
    """
    def __init__(self, ResNet_npy_path=None, trainable=True, open_tensorboard=False, dropout=0.8):
        if ResNet_npy_path is not None:
            self.data_dict = np.load(ResNet_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.dropout = dropout
        self.is_training = True
    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - configs['VGG_MEAN'][0],
            green - configs['VGG_MEAN'][1],
            red - configs['VGG_MEAN'][2],
        ])
        print(bgr.get_shape().as_list())
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        self.bgr = bgr
        self.conv1 = self.conv_layer(self.bgr, 7, 3, 64, 2, "conv1")# 112*112
        self.pool1 = self.max_pool(self.conv1, 3, 2, "pool1")# 56*56 * 64
        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "res2a", True)# 56*56
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "res2b")# 56*56
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "res2c")# 56*56

        self.pool2 = self.max_pool(self.block1_3, 2, 2, "pool2")# 56*56
        self.block2_1 = self.res_block_3_layers(self.pool2, [128, 128, 512], "res3a", True)# 28*28
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "res3b")# 28*28
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "res3c")# 28*28
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "res3d")# 28*28

        self.pool3 = self.max_pool(self.block2_4, 2, 2, "pool3")# 28*28
        self.block3_1 = self.res_block_3_layers(self.pool3, [256, 256, 1024], "res4a", True)# 14*14
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "res4b")# 14*14
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "res4c")# 14*14
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "res4d")# 14*14
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "res4e")# 14*14
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "res4f")# 14*14
        #[None 7 7 512]
        self.pool4 = self.max_pool(self.block3_6, 2, 2, "pool4")# 14*14
        self.block4_1 = self.res_block_3_layers(self.pool4, [512, 512, 2048], "res5a", True)# 7*7
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "res5b")# 7*7
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "res5c")# 7*7
        # upsample layer begins
        self.deconv1_1 = self.deconv_bn_relu(self.block4_3, name = 'deconv_1',kernel_size = 3, output_channels = 1024,
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 14*14
        self.conv1_1d = 
        self.deconv2_2 = self.deconv_bn_relu(self.deconv_1, name = 'deconv_2',kernel_size = 3, output_channels = 512,
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 28*28
        self.deconv3_3 = self.deconv_bn_relu(self.deconv_2, name = 'deconv_3',kernel_size = 3, output_channels = 256,
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 56*56
        self.deconv4_4 = self.deconv_bn_relu(self.deconv_3, name = 'deconv_4',kernel_size = 3, output_channels = 128,
                        initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 112*112
        self.deconv5_5 = self.deconv_bn_relu(self.deconv_4, name = 'deconv_5',kernel_size = 3, output_channels = 64,
                initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 224*224

        # self.final_layer = self.conv_la        self.deconv_1 = self.deconv_bn_relu(self.block4_3, name = 'deconv_1',kernel_size = 3, output_channels = 1024,
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 14*14yer(bottom = self.deconv_5, kernal_size = 1, in_channels = 64, out_channels = 3, stride = 1, name = 'final_layer')
        self.final_layer = self.conv_bn_relu(bottom = self.deconv_5, name = 'final_layer', kernel_size = 1, output_channels = 3, initializer =tf.contrib.layers.variance_scaling_initializer(), bn = False, training = self.is_training, relu=False)
        # self.pool5 = self.avg_pool(self.block4_3, 7, 1, "pool5")
        #self.fc0 = self.fc_layer(self.pool5, 2048, 1024, "fc0")
        #self.relu1 = tf.nn.relu(self.fc0)
        #if train_mode is not None:
        #    self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        #elif self.trainable:
        #    self.relu1 = tf.nn.dropout(self.relu1, self.dropout)

        self.y_soft = tf.nn.softmax(self.final_layer)
        self.logits = tf.reshape(self.final_layer, (-1, 3))
        print(self.logits)
        self.predicted = tf.argmax(self.final_layer, axis = 3)
        print(self.predicted.get_shape().as_list())
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits, name=None)


        # self.loss = tf.reduce_mean(cross_entropy, name = 'xcross_entropy')
        # if(last_layer_type == "sigmoid"):
	       #  self.prob = tf.nn.sigmoid(self.fc1, name="prob")
        # elif(last_layer_type == "softmax"):
        # 	self.prob = tf.nn.softmax(self.fc1, name="prob")

        self.data_dict = None
        return self.predicted


    def conv_bn_relu(self, bottom, kernel_size, out_channels, stride, name, train_mode):
        input_filter = bottom.get_shape().as_list()[-1]
        conv = self.conv_layer(bottom = bottom, kernel_size = kernel_size, in_channels = input_filter, 
            out_channels = out_channels, stride = 1, name)
        norm = tf.layers.batch_normalization(inputs=conv, axis = 3, 
                        momentum=configs['_BATCH_NORM_DECAY'],epsilon=configs['_BATCH_NORM_EPSILON'], 
                        center=True, scale=True, training=self.is_training, fused=True)
        relu = tf.nn.relu(norm)
        return relu


    def res_block_3_layers(self, bottom, channel_list, name, change_dimension = False):
        if (change_dimension):
            block_conv_input = self.conv_layer(bottom = bottom, kernal_size = 1, in_channels = bottom.get_shape().as_list()[-1],
                                               out_channels = channel_list[2], stride = 1, name = name + "_branch1")
        else:
            block_conv_input = bottom
        input_filter = bottom.get_shape().as_list()[-1]
        block_conv_1 = self.conv_layer(bottom, 1, input_filter, channel_list[0], 1, name + "_branch2a")
        block_norm_1 = tf.layers.batch_normalization(inputs=block_conv_1, axis = 3, momentum=configs['_BATCH_NORM_DECAY'], epsilon=configs['_BATCH_NORM_EPSILON'], center=True, scale=True, training=self.is_training, fused=True)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_branch2b")
        block_norm_2 = tf.layers.batch_normalization(inputs=block_conv_2, axis = 3, momentum=configs['_BATCH_NORM_DECAY'], epsilon=configs['_BATCH_NORM_EPSILON'], center=True, scale=True, training=self.is_training, fused=True)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_branch2c")
        block_res = tf.add(block_conv_input, block_conv_3)
        relu = tf.nn.relu(block_res)

        return relu


    def ResNet_Block(self, bottom, channel_list, name):
        input_filter = bottom.get_shape().as_list()[-1]
        conv_bn_relu1 = self.Conv_Bn_Relu(name = name + '_branch2a', bottom = bottom, output_channels = channel_list[0], kernel_size = 1, stride = 1, relu = True, bn = True)
        conv_bn_relu2 = self.Conv_Bn_Relu(name = name + '_branch2b', bottom = conv_bn_relu1, output_channels = channel_list[1], kernel_size = 3, stride = 1, relu = True, bn = True)
        block_conv_3 = self.conv_layer(conv_bn_relu2, 1, channel_list[1], channel_list[2], 1, name + '_branch2c')
        block_res = tf.add(bottom, block_conv_3)
        relu = tf.nn.relu(block_res)
        return relu


    def Dense_Block(self, bottom, name, stride = 1):
        """
        dense block composed with a down channel convlution with fiter_size =1
        and a up channel convolution with fiter_size = 3
        """
        input_channels = bottom.get_shape().as_list()[-1]
        dense_block_1 = self.BN_Relu_Conv(name + '_x1', bottom, input_channels = input_channels, 
                                            output_channels = K*4, kernel_size = 1, stride = 1)
        dense_block_2 = self.BN_Relu_Conv(name + '_x2', dense_block_1, input_channels = K*4, 
                                            output_channels = K, kernel_size = 3, stride = 1)
        dense_block = tf.concat([bottom, dense_block_2], axis = 3)
        print('Dense_Block layer {0} -> {1}'.format(bottom.get_shape().as_list(),dense_block.get_shape().as_list()))
        return dense_block

    def BN_Relu_Conv(self, name, bottom, input_channels, output_channels, kernel_size, stride = 1):
       batch_norm_scale = self.batch_norm_layer(name, bottom,phase_train = self.train_mode)
       relu = tf.nn.relu(batch_norm_scale)
       conv = self.conv_layer(bottom = relu, kernel_size = kernel_size, in_channels = input_channels, 
                                out_channels = output_channels, stride = stride, name = name)
       return conv


    def Conv_Bn_Relu(self, name, bottom, output_channels, kernel_size, stride = 1, relu = True, bn = True):

        input_channels = bottom.get_shape().as_list()[-1]
        conv_layer = self.conv_layer(bottom = bottom, kernel_size = kernel_size, in_channels = input_channels, 
            out_channels = output_channels, stride = stride, regularizer=tf.contrib.layers.l2_regularizer(0.0005) ,name = name)

        if bn == True:
            batch_norm_scale = self.batch_norm_layer(name = name, bottom = conv_layer, phase_train = self.train_mode)
        else:
            batch_norm_scale = conv_layer

        if relu == True:
            relu_layer = tf.nn.relu(batch_norm_scale)
        else:
            relu_layer = batch_norm_scale

        return relu_layer
        
    def avg_pool(self,bottom, kernel_size = 2, stride = 2, name = "avg"):
        avg_pool = tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
        print('avg_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),avg_pool.get_shape().as_list()))
        return avg_pool

    def max_pool(self,bottom, kernel_size = 3, stride = 2, name = "max"):
        max_pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
        print('max_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),max_pool.get_shape().as_list()))
        return max_pool

    def conv_layer(self, bottom, kernel_size, in_channels, out_channels, stride, name, regularizer = None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernel_size, in_channels, out_channels, name, regularizer = regularizer)
            conv = tf.nn.conv2d(bottom, filt, [1,stride,stride,1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            tf.summary.histogram('weight', filt)
            tf.summary.histogram('bias', conv_biases)

            return bias

    def batch_norm_layer(self, name, bottom, phase_train, decay=0.5):
        """
        glabal batch norm with input [batch_size height width channel]
        """
        n_out = bottom.get_shape().as_list()[-1]
        #restore the stored moving_mean moving_variance, beta, gamma if use pretrained model
        moving_mean,moving_variance,gamma,beta = self.get_batchnorm_var(n_out, name + '_bn')

        def mean_var_with_update():
            #if train model updata the moving mean and moving variance
            mean, variance = tf.nn.moments(bottom, [0,1,2], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        # if test eval model use the moving restored moving mean and moving_variance.
        mean, variance = tf.cond(phase_train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        return tf.nn.batch_normalization(bottom, mean, variance, beta, gamma, configs['_BATCH_NORM_EPSILON'])

    def deconv_bn_relu(self, bottom, name, kernel_size, output_channels, stride = 2, bn=False, training=False, relu=True):
        deconv_layer = self.deconv_layer(bottom, name, output_channels, kernel_size, stride, regularizer=None)
        if bn:
            deconv_layer = self.batch_norm_layer(name, bottom = deconv_layer, phase_train = self.train_mode)
        if relu:
            deconv_layer = tf.nn.relu(deconv_layer, name=name)

        print('Deconv layer {0} -> {1}'.format(bottom.get_shape().as_list(), deconv_layer.get_shape().as_list()))

        return deconv_layer

    def deconv_layer(self, bottom, name, output_channels, kernel_size, stride, regularizer=None):
        input_shape = bottom.get_shape().as_list()
        output_shape = [input_shape[0], input_shape[1]*stride, input_shape[2]*stride, output_channels]
        kernel_shape = [kernel_size, kernel_size, output_channels, input_shape[-1]]

        initial_weights = tf.truncated_normal(shape = kernel_shape, mean = 0, stddev = 1 / math.sqrt(float(kernel_size * kernel_size)))
        weights = self.get_var(initial_value = initial_weights, name =  name, idx = 'weights',  var_name = name + "_weights")
        initial_biases = tf.truncated_normal([output_channels], 0.0, 1.0)
        biases = self.get_var(initial_value = initial_biases, name =  name, idx = 'biases',  var_name = name + "_biases")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, [1, stride, stride, 1], padding='SAME')
        deconv_layer = tf.nn.bias_add(deconv, biases)

        return deconv_layer

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            tf.summary.histogram('weight', weights)
            tf.summary.histogram('bias', biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, regularizer = None):
        """get the conv weights and baises in pretrained model if not initialize with ramdom numbers.
        """
        trunc_stddev = self.variance_scaling_initializer(shape = [filter_size, filter_size, in_channels, out_channels],
                                         mode = 'FAN_AVG', uniform = False, factor = 2)
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, stddev = trunc_stddev)
        filters = self.get_var(initial_value = initial_value, name =  name, idx = 'weights',  var_name = "_filters",regularizer = regularizer)
        
        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value = initial_value, name = name, idx = 'biases', var_name = "_biases")
        
        return filters, biases

    def variance_scaling_initializer(self, shape, mode = 'FAN_AVG', uniform = False, factor = 2):
        """
        initialization
        """
        fin_in = shape[-2]*shape[0]*shape[1]
        fin_out = shape[-1]*shape[0]*shape[1]
        if mode == "FAN_IN":
            n = fin_in
        elif mode == "FAN_OUT":
            n = fin_out
        elif mode == "FAN_AVG":
            n = (fin_in+ fin_out)/2
        if uniform ==False:
            
            return math.sqrt(1.3*factor/n)
        else:
            return math.sqrt(3.0 * factor / n)



    def get_batchnorm_var(self, n_out, name):
        """batch normal parameters:
        1. mean(batch_var) 2. variance(batch_mean) 3. scale(gamma) 4. offset(beta)
        """
        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        beta = self.get_var(initial_value = init_beta, name =  name, idx = 'offset',  var_name = name + "_offset")

        init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float32)
        gamma = self.get_var(initial_value = init_gamma, name =  name, idx = 'scale',  var_name = name + "_scale")

        init_variance = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
        variance = self.get_var(initial_value = init_variance, name =  name, idx = 'variance',  var_name = name + "_variance")

        init_mean = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        mean = self.get_var(initial_value = init_mean, name =  name, idx = 'mean',  var_name = name + "_mean")

        return mean, variance, gamma, beta


    def get_fc_var(self, in_size, out_size, name):
        """
        in_size : number of input feature size
        out_size : number of output feature size
        name : block_layer name
        """
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev = 1 / math.sqrt(float(in_size)))
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases


    def get_var(self, initial_value, name, idx, var_name, regularizer = None, trainable=True):
        """
        detect if the pretrained model has the variable if not creat one as expected
        """
        if self.data_dict is not None and name in self.data_dict.keys() and idx in self.data_dict[name]:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            
        if self.trainable:
            var = tf.get_variable(name = var_name, initializer=value, trainable=trainable, regularizer = regularizer)
            # tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        
        return var


    def save_npy(self, sess, npy_path="./model_saved.npy"):
        """
        Save this model into a npy file
        """
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):

        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    def mean_iou(self, logits, labels):
        #logits imported from the last layer of the net[]
        #
        return logits
