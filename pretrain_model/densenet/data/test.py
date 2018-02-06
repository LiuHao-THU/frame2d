import numpy as np
pretrained_dir = '../densenet121.npy'
data_dict = np.load(pretrained_dir, encoding='latin1').item()
print(data_dict)
import tensorflow as tf

#t = tf.constant([[1, 2, 3], [4, 5, 6]])
#paddings = tf.constant([[1, 1,], [2, 2]])
#results = tf.pad(t, paddings, "CONSTANT")


# pretrained_dir_1 = '20.npy'
# data_dict_1 = np.load(pretrained_dir_1, encoding='latin1').item()

# pretrained_dir_2 = '21.npy'
# data_dict_2 = np.load(pretrained_dir_2, encoding='latin1').item()

# t = tf.zeros([10,3,3,1])
# paddings = [[0,0],[1,1],[1,1],[0,0]]
# z = tf.pad(t,paddings , "CONSTANT")