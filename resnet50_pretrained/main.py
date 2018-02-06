"""
Used to train ResNet
"""

# 5e-07, softmax, Adam, f0,relu, f1                    # 001
# 1e-06, sigmoid, Adam, f0,relu, f1                    # 002
# 1e-06, weight_sigmoid, 10, Adam, f0, relu, f1        # 003

# 1e-05, weight_sigmoid, 1, Adam, simple        # 00.npy
# 1e-05, weight_sigmoid, 5, Adam, train         # 01.npy
# 5e-06, weight_sigmoid, 5, Adam, train         # 02.npy end at epoch 40, learning rate 0.05 * 10^-6,

import os
import math
import time
import tensorflow as tf
# from ..model.dense_linknet import Dense_LinkNet as model
# import dense_linknet.Dense_Linknets as model
# from .dense_linknet import Dense_Linknet
import numpy as np
import scipy.io as scio
from scipy import misc
# from utils import *
from .configs import configs
from ..dataset import read_data
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]=configs['GPU']
# print(configs)
from .resnet50 import ResNet

read = read_data.read_data(root_dir = configs['root_dir'], save_dir = configs['save_dir'], image_size = configs['image_size'], per = configs['per'], configs = configs)
if configs['saved_npy'] == False:
    read.save_data()
#read data
read.build_train_input()
read.build_test_input()

with tf.Session() as sess:
    with tf.variable_scope("model") as scope:
        images = tf.placeholder(tf.float32, shape = [configs["batch_size"], configs['image_size'], configs['image_size'], configs['channel']])
        labels = tf.placeholder(tf.int32, shape = [configs["batch_size"], configs['image_size'], configs['image_size'], 1])
        eval_images = tf.placeholder(tf.float32, shape = [len(read.test_images), configs['image_size'], configs['image_size'], configs['channel']])


        # labels = tf.placeholder(tf.float32, shape = [configs["batch_size"], LABELSNUM])
        train_mode = tf.placeholder(tf.bool)

        # build resnet model
        # train_mode = True
        # 
        # with tf.device(configs['dev']):
        model = ResNet(npy_path = configs['model_path'])
        model.build(rgb = images, train_mode=True,label_num = 3)
        num_minibatches = int(read.train_num/ configs['batch_size'])

        # cost function
        
        with tf.name_scope("cost"):
            if(configs['final_layer_type'] == "sigmoid"):
                loss = tf.nn.weighted_cross_entropy_with_logits(logits = model.logits, targets = labels, pos_weight = 5.0)
            elif(configs['final_layer_type'] == "softmax"):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits = tf.reshape(model.final_layer,[configs['batch_size'],-1]), labels = label_one_hot)
            elif(configs['final_layer_type'] == "softmax_sparse"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model.logits, labels = tf.reshape(labels, [configs['batch_size'],-1]))
            cost = tf.reduce_mean(loss)
        with tf.name_scope("train"):
            global_steps = tf.Variable(0,trainable=False)
            learning_rate = tf.train.exponential_decay(configs['learning_rate_orig'], global_steps, num_minibatches*2, 0.99, staircase = True)
            # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            # train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        sess.run(tf.global_variables_initializer())
        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        # print(model.get_var_count())

        # if(configs['tensorboard_on']):
        #     merged_summary = tf.summary.merge_all()
        #     writer = tf.summary.FileWriter(configs['tensorboard_dir'])
        #     writer.add_graph(sess.graph)

        # used in tensorboard to count record times
        summary_times = 0
        saver = tf.train.Saver(max_to_keep = 2)

        for epoch in range(configs['epoch']):
            print("Start Epoch %i" % (epoch + 1))
            batch_index = 0
            minibatch_cost = 0
            for batch in read.minibatches_train(batch_size = configs['batch_size'], shuffle = True):
                # get train examples from each mini batch
                images_batch,labels_batch = batch
                #show images plt
                # change learning rate
                learning_rate = tf.train.exponential_decay(configs['learning_rate_orig'], epoch * num_minibatches + batch_index, num_minibatches, 0.96, staircase = True)
                

                sess.run(global_steps.assign(epoch * num_minibatches + batch_index))
                # Training and calculating cost
                model.set_is_training(True)
                temp_cost, _ ,c= sess.run([cost, train, model.y_soft], feed_dict={images: images_batch, labels: labels_batch})
                minibatch_cost += np.sum(temp_cost)
                batch_index += 1

                # if(configs['tensorboard_on']) and (batch_index % configs['tensorboard_refresh'] == 0):
                #     s = sess.run(merged_summary, feed_dict={images: images_batch, labels: labels_batch, train_mode: False})
                #     writer.add_summary(s, summary_times)
                #     summary_times = summary_times + 1
                #     # record cost in tensorflow
                #     tf.summary.scalar('cost', temp_cost)
                    #tf.summary.image('input', minibatch_X, 10)
                
                if (batch_index % 1 == 0):
                    print("Epoch %i Batch %i Batch Cost %f Learning_rate %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate)))

                # record examples to monitoring the training process
                # if((batch_index % 50 == 1) and (epoch % 1 == 0)):
                    # model.set_is_training(False)
                    # z2 = sess.run([model], feed_dict={eval_images: read.test_images, train_mode: False})
                    # scio.savemat("./tmp/name_%i_%i.mat"%(epoch,batch_index), {'fc1': fc1, 'prob':prob, 'label':minibatch_Y})
                # if((batch_index + 1) % configs['save_frequency'] == 0):
                    # save_path = saver.save(sess, configs['checkpoint_dir'])
                    # model.save_npy(sess, configs['checkpoint_dir']+'%i.npy' % (epoch + 1))
            # print total cost of this epoch
            print("End Epoch %i" % epoch, "Total cost of Epoch %f" % minibatch_cost)
            # save model
            saver.save(sess, configs['checkpoint_dir']+ '/' +str(epoch)+'_model.ckpt', global_step=epoch * num_minibatches + batch_index)

