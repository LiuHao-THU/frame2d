import tensorflow as tf
from six.moves import range
from .configs import configs
from .dnfss import DNFSS
from ..dataset.read_data import read_data
import time
import matplotlib.pyplot as plt
import os
#read data
os.environ["CUDA_VISIBLE_DEVICES"]=configs['GPU']


read = read_data(root_dir = configs['root_dir'], save_dir = configs['save_dir'], image_size = configs['image_size'], per = configs['per'], configs = configs)
if configs['saved_npy'] == False:
	read.save_data()
#read data
read.build_train_input()
read.build_test_input()



#***************************************train the data***************************************
#1. restore the session
x = tf.placeholder(tf.float32, shape=(configs['batch_size'], configs['image_size'],configs['image_size'], configs['channel'])) #train images
y = tf.placeholder(tf.int64, shape=(configs['batch_size'], configs['image_size'], configs['image_size'], 1)) #train labels
image = tf.placeholder(tf.float32, shape=(configs['batch_size'],configs['image_size'],configs['image_size'],configs['channel'])) #test images
rate = tf.placeholder(tf.float32, shape=[])  #initial learning rate
net = DNFSS(x = x, y = y, image = image, rate = rate)
# net.inference()
net.build()
saver = tf.train.Saver(max_to_keep = 5)
config = tf.ConfigProto(allow_soft_placement = True)
session = tf.Session(config = config)
session.run(tf.global_variables_initializer())
iter = 0
for i in range(configs['epoch']):
	for batch in read.minibatches_train(batch_size = configs['batch_size'], shuffle = True):

		images,labels = batch

		#imshow images
		start = time.time()
		_,loss,xx = session.run([net.train_step,net.loss,net.xx],feed_dict={x: images, y: labels, rate: configs['lrn_rate']})

		print('this is the loss function of the data %f', loss)
		iter = iter + 1
	saver.save(session, configs['checkpoint_dir']+'/'+ str(i)+'.ckpt', global_step=i)
	print('Model {} saved'.format(i))
