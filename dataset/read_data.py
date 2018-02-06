# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:14:22 2017

@author: LIUHAO -THU
"""
#read data
"""
root_dir
	---index_file
		---train
		---label
		---info
			read_all_data
    we do not need info data when train the network
"""
import os
import scipy
import numpy as np
from random import randint
import matplotlib.pyplot as plt
# from configs import configs
from .augmentation import augmentation
import tensorflow as tf
import scipy.misc
import cv2

class read_data(object):
    """docstring for ClassName"""
    def __init__(self, root_dir, save_dir, image_size, per, configs):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.image_size = image_size
        self.per = per
        self.configs = configs
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
    def read_train_and_label(self,sub_dir):
        train_dir = os.path.join(sub_dir,'train')
        label_dir = os.path.join(sub_dir,'label')
        train_dir_list = os.listdir(train_dir)
        label_dir_list = os.listdir(label_dir)
        img_train = []
        img_label = []
        for i in train_dir_list:
            # img_train.append(scipy.misc.imresize(plt.imread(os.path.join(train_dir,i)), size = [self.image_size, self.image_size], interp = 'bilinear'))
            img_train.append(cv2.resize(plt.imread(os.path.join(train_dir,i)), dsize = (self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR))
        for i in label_dir_list:
            img_label.append(cv2.resize(plt.imread(os.path.join(label_dir,i)), dsize = (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST ))
            # img_label.append(scipy.misc.imresize(plt.imread(os.path.join(label_dir,i)), size = (self.image_size, self.image_size), interp = 'nearest'))
        return img_train,img_label

    def save_data(self):
        all_dir = os.listdir(os.path.join(self.dir_path,self.root_dir))
        train_data = []
        label_data = []
        iter = 0
        print(os.path.join(self.dir_path,self.root_dir))
        for i in all_dir:
            iter = iter + 1
            print(iter)
            sub_dir = os.path.join(self.dir_path,self.root_dir,i)
            img_train,img_label = self.read_train_and_label(sub_dir)
            #save data as .npy format
            [train_data.append(np.array(np.resize(np.stack((x,x,x), axis = 2), [self.image_size, self.image_size,3]))) for x in img_train]
            [label_data.append(np.array(np.resize(x, [self.image_size, self.image_size,1]))) for x in img_label]
#            train_data.append(img_train)
        # print(len(train_data))
#            label_data.append(img_label)
            # 
        argumentation_class = augmentation(train_data, label_data, self.configs)
        train_data, label_data = argumentation_class.augmentation_final(raw_images = self.configs['raw_images'], horizontal_flip_num = self.configs['horizontal_flip_num'], vertical_flip_num = self.configs['vertical_flip_num'],
                                                random_rotate_num = self.configs['random_rotate_num'], random_crop_num = self.configs['random_crop_num'],center_crop_num = self.configs['center_crop_num'],
												slide_crop_num = self.configs['slide_crop_num'],slide_crop_old_num = self.configs['slide_crop_old_num'])

        np.save(os.path.join(self.dir_path, self.save_dir, self.configs['imgs_train']), train_data[0:round(len(train_data)*self.per)])
        np.save(os.path.join(self.dir_path, self.save_dir, self.configs['imgs_label']), label_data[0:round(len(train_data)*self.per)])
        np.save(os.path.join(self.dir_path, self.save_dir, self.configs['imgs_train_test']), train_data[round(len(train_data)*self.per):])
        np.save(os.path.join(self.dir_path, self.save_dir, self.configs['imgs_label_test']), label_data[round(len(train_data)*self.per):])

        #scipy.io.savemat(os.path.join(self.dir_path, self.save_dir,'imgs_label_test.mat'),mdict={'label_str':label_data[round(len(train_data)*self.per):]})

        return train_data, label_data

    def normalize(self, data):
        #normalize the data to 0---1
        return (data - data.mean())/data.std()

    def build_train_input(self):
        train_image_dir = os.path.join(self.dir_path, self.save_dir, self.configs['imgs_train'])
        train_label_dir = os.path.join(self.dir_path, self.save_dir, self.configs['imgs_label'])
        self.train_images = np.load(train_image_dir)
        # self.train_images = self.train_images/(np.max(self.train_images) - np.min(self.train_images))*255
        self.train_labels = np.load(train_label_dir)
        



        self.train_num = len(self.train_images)
        print('max_number of self.train_labels')
        print(self.train_labels.shape)
        # self.train_images = self.normalize(self.train_images)



            # , , center_crop_num = configs['center_crop_num'],
            #                                     slide_crop_num = configs['slide_crop_num'], slide_crop_old_num = configs['slide_crop_old_num']

    def build_test_input(self):
        test_image_dir = os.path.join(self.dir_path, self.save_dir, self.configs['imgs_train_test'])
        test_label_dir = os.path.join(self.dir_path, self.save_dir, self.configs['imgs_label_test'])
        self.test_images = np.load(test_image_dir)
        self.test_labels = np.load(test_label_dir)
        # self.test_images = self.test_images/(np.max(self.test_images) - np.min(self.test_images))*255
        return self.test_images,self.test_labels


    def minibatches_train(self, inputs=None, targets=None, batch_size=None, shuffle=False):
        """
        for batch in tl.iterate.minibatches(inputs=train_data, targets=train_labels, batch_size=batch_size, shuffle=True):
            images, labels = batch
        """
        assert len(self.train_images) == len(self.train_labels)
        if shuffle:
            indices = np.arange(len(self.train_images))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.train_images) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self.train_images[excerpt], self.train_labels[excerpt]

    def minibatches_eval(self, inputs=None, targets=None, batch_size=None, shuffle=False):
        """
        for batch in tl.iterate.minibatches(inputs=train_data, targets=train_labels, batch_size=batch_size, shuffle=True):
            images, labels = batch
        """
        assert len(self.test_images) == len(self.test_labels)
        if shuffle:
            indices = np.arange(len(self.test_images))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.test_images) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self.test_images[excerpt], self.test_labels[excerpt]

    def shuffle_data_index(self, batch_size):
        """shuffle the image in every epoch"""
        self.dataset_size = np.shape(self.train_images)[0]
        data_index = np.arange(self.dataset_size)
        np.random.shuffle(data_index)
        self.max_batch = self.dataset_size//batch_size
        #generate random data from 0--dataset_size - 1
        random_index = np.random.randint(low = 0, high = self.dataset_size-1, size = batch_size*(self.max_batch+1) - self.dataset_size)
        self.data_index = np.concatenate((data_index, random_index), axis=0)


    def shuffle_train_data(self, batch_size, batch_iter):
        images   = np.empty([batch_size, self.image_size, self.image_size, 1],dtype=np.float32)
        labels = np.empty([batch_size, self.image_size, self.image_size, 1],dtype=np.float32)
        """return shuffled data when given batch_iter"""
        if (batch_iter+1)*batch_size < self.dataset_size:

            for i in range(batch_size):
                images[i] = self.train_images[self.data_index[batch_iter*batch_size + i]]
                labels[i] = self.train_labels[self.data_index[batch_iter*batch_size + i]]
            return images,labels,False
        else:
            for i in range(batch_size):
                images[i] = self.train_images[self.data_index[batch_iter*batch_size + i]]
                labels[i] = self.train_labels[self.data_index[batch_iter*batch_size + i]]
            return images,labels,True
