"""
Time: 2017.12.27
Author: LiuHao
Institution:THU
"""
"""
dense net name 
conv1/bn + conv1/scale + conv1//relu + conv1 + pool2(max)
conv2_1/x1.2(/bn+/scale+relu+conv)
...
conv2_6/x1.2(/bn+/scale+relu+conv) concat
conv2_blk/bn + conv2_blk/scale + conv2_blk/relu + conv2_blk + pool2(avg)
conv3_1/x1.2(/bn+/scale+relu+conv)
...
conv3_12/x1.2(/bn+/scale+relu+conv) concat
conv3_blk/bn + conv3_blk/scale + conv3_blk/relu + conv3_blk + pool3(avg)
conv4_1/x1.2(/bn+/scale+relu+conv)
...
conv4_2/x1.2(/bn+/scale+relu+conv) concat
conv4_blk/bn + conv4_blk/scale + conv4_blk/relu + conv4_blk + pool4(avg)
conv5_1/x1.2(/bn+/scale+relu+conv)
...
conv5_16/x1.2(/bn+/scale+relu+conv) concat
"""
# input dimension batch_size * 224 * 224 * channels
# out_size batch_size * 8 * 8 * 1024

"""
This code is the first edition of the dense net for segmentation 
The encode part use the pretrained network from the "https://github.com/shicai/DenseNet-Caffe"
we use transpose convolution and deconv operation 
"""


import math
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.python.ops import random_ops
import six
from tensorflow.python.training.moving_averages import assign_moving_average
from .configs import configs
K = 32
middle_layer = K * 4

class Dense_Linknet:
	# some properties
    """
    Initialize function
    """
    def __init__(self, npy_path=None, trainable=True, open_tensorboard=False, dropout=0.8):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.dropout = dropout

    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        self.train_mode = train_mode
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
        self.conv1 = self.Conv_Bn_Relu(name = 'conv1', bottom = bgr, output_channels = 64, kernel_size = 7, stride = 2)
        self.pool1 = self.max_pool(bottom = self.conv1, kernel_size = 3, stride = 2, name = "pool1")# 56*56* 64
        self.block1_1 = self.Dense_Block(self.pool1, name = "conv2_1", stride = 1)# 56*56*256
        self.block1_2 = self.Dense_Block(self.block1_1, name = "conv2_2", stride = 1)# 56*56*256
        self.block1_3 = self.Dense_Block(self.block1_2, name = "conv2_3", stride = 1)# 56*56*256
        self.block1_4 = self.Dense_Block(self.block1_3, name = "conv2_4", stride = 1)# 56*56*256
        self.block1_5 = self.Dense_Block(self.block1_4, name = "conv2_5", stride = 1)# 56*56*256
        self.block1_6 = self.Dense_Block(self.block1_5, name = "conv2_6", stride = 1)# 56*56*256
        #stage3
        self. blk2 = self.BN_Relu_Conv("conv2_blk", self.block1_6, input_channels = self.block1_6.get_shape().as_list()[-1], 
        								output_channels =self.block1_6.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)

        self.pool2 = self.avg_pool(bottom = self. blk2, kernel_size = 2, stride = 2, name = "pool2")
        self.block2_1 = self.Dense_Block(self.pool2, name = "conv3_1", stride = 1)# 28*28*512
        self.block2_2 = self.Dense_Block(self.block2_1, name = "conv3_2", stride = 1)# 28*28*512
        self.block2_3 = self.Dense_Block(self.block2_2, name = "conv3_3", stride = 1)# 28*28*512
        self.block2_4 = self.Dense_Block(self.block2_3, name = "conv3_4", stride = 1)# 28*28*512
        self.block2_5 = self.Dense_Block(self.block2_4, name = "conv3_5", stride = 1)# 28*28*512
        self.block2_6 = self.Dense_Block(self.block2_5, name = "conv3_6", stride = 1)# 28*28*512
        self.block2_7 = self.Dense_Block(self.block2_6, name = "conv3_7", stride = 1)# 28*28*512
        self.block2_8 = self.Dense_Block(self.block2_7, name = "conv3_8", stride = 1)# 28*28*512
        self.block2_9 = self.Dense_Block(self.block2_8, name = "conv3_9", stride = 1)# 28*28*512
        self.block2_10 = self.Dense_Block(self.block2_9, name = "conv3_10", stride = 1)# 28*28*512
        self.block2_11 = self.Dense_Block(self.block2_10, name = "conv3_11", stride = 1)# 28*28*512
        self.block2_12 = self.Dense_Block(self.block2_11, name = "conv3_12", stride = 1)# 28*28*512
        self. blk3 = self.BN_Relu_Conv("conv3_blk", self.block2_12, input_channels = self.block2_12.get_shape().as_list()[-1], 
        								output_channels =self.block2_12.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)
        self.pool3 = self.avg_pool(bottom = self. blk3, kernel_size = 2, stride = 2, name = "pool3")
        #stage4
        self.block3_1 = self.Dense_Block(self.pool3, name = "conv4_1", stride = 1)# 14*14*1024
        self.block3_2 = self.Dense_Block(self.block3_1, name = "conv4_2", stride = 1)# 14*14*1024
        self.block3_3 = self.Dense_Block(self.block3_2, name = "conv4_3", stride = 1)# 14*14*1024
        self.block3_4 = self.Dense_Block(self.block3_3, name = "conv4_4", stride = 1)# 14*14*1024
        self.block3_5 = self.Dense_Block(self.block3_4, name = "conv4_5", stride = 1)# 14*14*1024
        self.block3_6 = self.Dense_Block(self.block3_5, name = "conv4_6", stride = 1)# 14*14*1024
        self.block3_7 = self.Dense_Block(self.block3_6, name = "conv4_7", stride = 1)# 14*14*1024
        self.block3_8 = self.Dense_Block(self.block3_7, name = "conv4_8", stride = 1)# 14*14*1024
        self.block3_9 = self.Dense_Block(self.block3_8, name = "conv4_9", stride = 1)# 14*14*1024
        self.block3_10 = self.Dense_Block(self.block3_9, name = "conv4_10", stride = 1)# 14*14*1024
        self.block3_11 = self.Dense_Block(self.block3_10, name = "conv4_11", stride = 1)# 14*14*1024
        self.block3_12 = self.Dense_Block(self.block3_11, name = "conv4_12", stride = 1)# 14*14*1024
        self.block3_13 = self.Dense_Block(self.block3_12, name = "conv4_13", stride = 1)# 14*14*1024
        self.block3_14 = self.Dense_Block(self.block3_13, name = "conv4_14", stride = 1)# 14*14*1024
        self.block3_15 = self.Dense_Block(self.block3_14, name = "conv4_15", stride = 1)# 14*14*1024
        self.block3_16 = self.Dense_Block(self.block3_15, name = "conv4_16", stride = 1)# 14*14*1024
        self.block3_17 = self.Dense_Block(self.block3_16, name = "conv4_17", stride = 1)# 14*14*1024
        self.block3_18 = self.Dense_Block(self.block3_17, name = "conv4_18", stride = 1)# 14*14*1024
        self.block3_19 = self.Dense_Block(self.block3_18, name = "conv4_19", stride = 1)# 14*14*1024
        self.block3_20 = self.Dense_Block(self.block3_19, name = "conv4_20", stride = 1)# 14*14*1024
        self.block3_21 = self.Dense_Block(self.block3_20, name = "conv4_21", stride = 1)# 14*14*1024
        self.block3_22 = self.Dense_Block(self.block3_21, name = "conv4_22", stride = 1)# 14*14*1024
        self.block3_23 = self.Dense_Block(self.block3_22, name = "conv4_23", stride = 1)# 14*14*1024
        self.block3_24 = self.Dense_Block(self.block3_23, name = "conv4_24", stride = 1)# 14*14*1024
        self. blk4 = self.BN_Relu_Conv("conv4_blk", self.block3_24, input_channels = self.block3_24.get_shape().as_list()[-1], 
        								output_channels =self.block3_24.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)
        self.pool4 = self.avg_pool(bottom = self. blk4, kernel_size = 2, stride = 2, name = "pool4")
        #stage5

        self.block4_1 = self.Dense_Block(self.pool4, name = "conv5_1", stride = 1)# 8*8*1024
        self.block4_2 = self.Dense_Block(self.block4_1, name = "conv5_2", stride = 1)# 8*8*1024
        self.block4_3 = self.Dense_Block(self.block4_2, name = "conv5_3", stride = 1)# 8*8*1024
        self.block4_4 = self.Dense_Block(self.block4_3, name = "conv5_4", stride = 1)# 8*8*1024
        self.block4_5 = self.Dense_Block(self.block4_4, name = "conv5_5", stride = 1)# 8*8*1024
        self.block4_6 = self.Dense_Block(self.block4_5, name = "conv5_6", stride = 1)# 8*8*1024
        self.block4_7 = self.Dense_Block(self.block4_6, name = "conv5_7", stride = 1)# 8*8*1024
        self.block4_8 = self.Dense_Block(self.block4_7, name = "conv5_8", stride = 1)# 8*8*1024
        self.block4_9 = self.Dense_Block(self.block4_8, name = "conv5_9", stride = 1)# 8*8*1024
        self.block4_10 = self.Dense_Block(self.block4_9, name = "conv5_10", stride = 1)# 8*8*1024
        self.block4_11 = self.Dense_Block(self.block4_10, name = "conv5_11", stride = 1)# 8*8*1024
        self.block4_12 = self.Dense_Block(self.block4_11, name = "conv5_12", stride = 1)# 8*8*1024
        self.block4_13 = self.Dense_Block(self.block4_12, name = "conv5_13", stride = 1)# 8*8*1024
        self.block4_14 = self.Dense_Block(self.block4_13, name = "conv5_14", stride = 1)# 8*8*1024
        self.block4_15 = self.Dense_Block(self.block4_14, name = "conv5_15", stride = 1)# 8*8*1024
        self.block4_16 = self.Dense_Block(self.block4_15, name = "conv5_16", stride = 1)# 8*8*1024
        # upsample layer begins



        # upsample layer begins
        self.deconv1_1 = self.deconv_bn_relu(self.block4_16, name = 'deconv1_1',kernel_size = 3, output_channels = 512, stride=2, bn=True, training=self.train_mode)# 14*14
        self.concat1_1 = tf.concat([self.deconv1_1, self.blk4], axis = 3)
        self.res1_0 = self.conv_layer(self.concat1_1, 1, 1024, 512, 1, name = 'res1_0')
        self.res1_1 = self.ResNet_Block(self.res1_0, channel_list = [256,256,512] ,name = "res1_1")# 14*14*512
        self.res1_2 = self.ResNet_Block(self.res1_1, channel_list = [256,256,512] ,name = "res1_2")# 14*14*512
        self.res1_3 = self.ResNet_Block(self.res1_2, channel_list = [256,256,512] ,name = "res1_3")# 14*14*512
        self.res1_4 = self.ResNet_Block(self.res1_3, channel_list = [256,256,512] ,name = "res1_4")# 14*14*512
        self.res1_5 = self.ResNet_Block(self.res1_4, channel_list = [256,256,512] ,name = "res1_5")# 14*14*512

        self.deconv2_1 = self.deconv_bn_relu(self.res1_5, name = 'deconv_2',kernel_size = 3, output_channels = 256, stride=2, bn=True, training=self.train_mode)# 28*28
        self.concat2_1 = tf.concat([self.deconv2_1, self.blk3], axis = 3)
        self.res2_0 = self.conv_layer(self.concat2_1, 1, 512, 256, 1, name = 'res2_0')
        self.res2_1 = self.ResNet_Block(self.res2_0, channel_list = [128,128,256] ,name = "res2_1")# 28*28*256
        self.res2_2 = self.ResNet_Block(self.res2_1, channel_list = [128,128,256] ,name = "res2_2")# 28*28*256
        self.res2_3 = self.ResNet_Block(self.res2_2, channel_list = [128,128,256] ,name = "res2_3")# 28*28*256
        self.res2_4 = self.ResNet_Block(self.res2_3, channel_list = [128,128,256] ,name = "res2_4")# 28*28*256
        self.res2_5 = self.ResNet_Block(self.res2_4, channel_list = [128,128,256] ,name = "res2_5")# 28*28*256

        self.deconv3_2 = self.deconv_bn_relu(self.res2_5, name = 'deconv_3',kernel_size = 3, output_channels = 128, stride=2, bn=True, training=self.train_mode)# 56*56
        self.concat3_1 = tf.concat([self.deconv3_2, self.blk2], axis = 3)
        self.res3_0 = self.conv_layer(self.concat3_1, 1, 256, 128, 1, name = 'res3_0')
        self.res3_1 = self.ResNet_Block(self.res3_0, channel_list = [64,64,128] ,name = "res3_1")# 56*56*128
        self.res3_2 = self.ResNet_Block(self.res3_1, channel_list = [64,64,128] ,name = "res3_2")# 56*56*128
        self.res3_3 = self.ResNet_Block(self.res3_2, channel_list = [64,64,128] ,name = "res3_3")# 56*56*128
        self.res3_4 = self.ResNet_Block(self.res3_3, channel_list = [64,64,128] ,name = "res3_4")# 56*56*128
        self.res3_5 = self.ResNet_Block(self.res3_4, channel_list = [64,64,128] ,name = "res3_5")# 56*56*128




        self.deconv4_3 = self.deconv_bn_relu(self.res3_5, name = 'deconv_4',kernel_size = 3, output_channels = 64, stride=2, bn=True, training=self.train_mode)# 112*112
        self.concat4_1 = tf.concat([self.deconv4_3, self.conv1], axis = 3)
        self.res4_0 = self.conv_layer(self.concat4_1, 1, 128, 64, 1, name = 'res4_0')
        self.res4_1 = self.ResNet_Block(self.res4_0, channel_list = [32,32,64] ,name = "res4_1")# 112*112*64
        self.res4_2 = self.ResNet_Block(self.res4_1, channel_list = [32,32,64] ,name = "res4_2")# 112*112*64
        self.res4_3 = self.ResNet_Block(self.res4_2, channel_list = [32,32,64] ,name = "res4_3")# 112*112*64
        self.res4_4 = self.ResNet_Block(self.res4_3, channel_list = [32,32,64] ,name = "res4_4")# 112*112*64
        self.res4_5 = self.ResNet_Block(self.res4_4, channel_list = [32,32,64] ,name = "res4_5")# 112*112*64

        self.deconv5_4 = self.deconv_bn_relu(self.res4_5, name = 'deconv_5',kernel_size = 3, output_channels = 32, stride=2, bn=True, training=self.train_mode)# 224*224

        print(self.deconv5_4)
        # self.concat5_1 = tf.concat([self.deconv5_4, self.convd], axis = 3)
        self.res5_0 = self.conv_layer(self.deconv5_4, 1, 32, 32, 1, name = 'res5_0')
        self.res5_1 = self.ResNet_Block(self.res5_0, channel_list = [16,16,32] ,name = "res5_1")# 112*112*64
        self.res5_2 = self.ResNet_Block(self.res5_1, channel_list = [16,16,32] ,name = "res5_2")# 112*112*64
        self.res5_3 = self.ResNet_Block(self.res5_2, channel_list = [16,16,32] ,name = "res5_3")# 112*112*64
        self.res5_4 = self.ResNet_Block(self.res5_3, channel_list = [16,16,32] ,name = "res5_4")# 112*112*64
        self.res5_5 = self.ResNet_Block(self.res5_4, channel_list = [16,16,32] ,name = "res5_5")# 112*112*64

        self.final_layer = self.Conv_Bn_Relu(name = 'final_layer', bottom = self.res5_5, output_channels = 21, kernel_size = 1, stride = 1, relu = False, bn = False)


        self.y_soft = tf.nn.softmax(self.final_layer)
        output_shape = self.final_layer.get_shape().as_list()
        self.logits = tf.reshape(self.final_layer,[output_shape[0],-1, 21])
        # self.logits = tf.reshape(self.final_layer, (-1, 3))
        self.pred = tf.argmax(self.y_soft, axis = 3)



    def ResNet_Block(self, bottom, channel_list, name):
        input_filter = bottom.get_shape().as_list()[-1]
        conv_bn_relu1 = self.Conv_Bn_Relu(name = name + '_branch2a', bottom = bottom, output_channels = channel_list[0], kernel_size = 1, stride = 1, relu = True, bn = True)
        conv_bn_relu2 = self.Conv_Bn_Relu(name = name + '_branch2b', bottom = conv_bn_relu1, output_channels = channel_list[1], kernel_size = 3, stride = 1, relu = True, bn = True)
        block_conv_3 = self.conv_layer(conv_bn_relu2, 1, channel_list[1], channel_list[2], 1, name + '_branch2c')
        relu = tf.nn.relu(block_conv_3)
        block_res = tf.add(bottom, relu)
        return block_res


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
