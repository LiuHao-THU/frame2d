# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:25:57 2017

@author: ANTIGEN
"""
"""
data argumentation demo
"""
import math
import numbers
import random
import numpy as np
# from scipy import misc
# from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from six.moves import range
# from configs import configs
import cv2
"""
random scale
random crop
horizontal/vertical flip
shift
rotation/reflection
noise
label shuffle
zoom
contrast
channel shift
pca????
"""



class augmentation(object):
    """docstring for ClassName"""
    def __init__(self, images, labels, configs):
        self.images = images
        self.labels = labels
        self.configs = configs
    def noise(self, images, labels, mean = 0, var = 0.1):
    	sigma = var**0.5
    	row,col,ch = images.shape
    	gauss = np.random.normal(mean,sigma,(row,col,ch))
    	return gauss + images, labels

    def _pad(self, images, labels, crop_size):
        h, w = images.shape[: 2]
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        images = np.pad(images, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        labels = np.pad(labels, ((0, pad_h), (0, pad_w)), 'constant')
        return images, labels

    def normalization(self, images):
        mean = images.mean()
        std = images.std()
        return (images-mean)/std

    def random_crop(self, images, labels, size):
        crop_size = [size, size]
        w,h = labels.shape
        th, tw = crop_size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return images[x1:x1+tw, y1:y1+th], labels[x1:x1 + tw, y1:y1 + th]

    def center_crop(self, images, labels, size):
        crop_size = [size, size]
        w, h = labels.shape
        th, tw = crop_size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return images[x1:x1+tw, y1:y1+th], labels[x1:x1 + tw, y1:y1 + th]

    def random_horizontally_flip(self, images, labels, randomly = True):
        if randomly == True:
            if random.random() < 0.5:
                return np.fliplr(images), np.fliplr(labels)
        else:
            return np.fliplr(images), np.fliplr(labels)

        return images, labels


    def random_vertically_flip(self, images, labels, randomly = True):
        if randomly == True:
            if random.random() < 0.5:
                return np.flipud(images), np.flipud(labels)
        else:
            return np.flipud(images), np.flipud(labels)
        return images, labels



    def random_rotate(self, images, labels, degree):
        rotate_degree = random.random()*2*degree -degree
        w,h = labels.shape
        rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotate_degree, 1)
        return cv2.warpAffine(images,  rotation_matrix,  (w,h), cv2.INTER_LINEAR), cv2.warpAffine(labels, rotation_matrix, (w,h), flags=cv2.INTER_NEAREST) #image use binear interpolation method labels nearest

    def scale(self, images, labels, size):
        """
        size represent the biggest size in width or height
        w,h reshape the images and labels size to w* h
        """
        # assert img.size == mask.size
        w, h = labels.shape
        if (w >= h and w == size) or (h >= w and h == size):
            return img, mask
            print('esdsa')
        if w > h:
            ow = size
            oh = int(size * h / w)


            return cv2.resize(images, dsize = (ow, oh), interpolation = cv2.INTER_LINEAR), cv2.resize(labels, dsize = (ow, oh), interpolation = cv2.INTER_NEAREST)
        # misc.imresize(images, [ow, oh], interp = 'bilinear'), misc.imresize(labels, [ow, oh], interp = 'nearest')
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(images, dsize = (ow, oh), interpolation = cv2.INTER_LINEAR),cv2.resize(labels, dsize = (ow, oh), interpolation = cv2.INTER_NEAREST)
        # misc.imresize(images, [ow, oh], interp = 'bilinear'), misc.imresize(labels, [ow, oh], interp = 'nearest')

    def random_sized_crop(self, images, labels, size, target_ratio = 0.5):
        raw_shape = labels.shape
        for attempt in range(10):
            area = labels.shape[0] * labels.shape[1]
            target_area = random.uniform(target_ratio, 1.0) * area
            distortion_ratio = random.uniform(0.5, 2)
            w = int(round(math.sqrt(target_area * distortion_ratio)))
            h = int(round(math.sqrt(target_area / distortion_ratio)))
            print(w,h)
            if random.random() < 0.5:
                w, h = h, w

            if w <= labels.shape[0] and h <= labels.shape[1]:
                x1 = random.randint(0, labels.shape[0] - w)
                y1 = random.randint(0, labels.shape[1] - h)
                #crop images
                images = images[x1:x1 + w, y1:y1 + h]
                labels = labels[x1:x1 + w, y1:y1 + h]
                # assert (images.shape == (w, h))
                return cv2.resize(images, dsize = (raw_shape[0], raw_shape[1]), interpolation = cv2.INTER_LINEAR),cv2.resize(labels, dsize = (raw_shape[0], raw_shape[1]), interpolation = cv2.INTER_NEAREST)
            # misc.imresize(images, [raw_shape[0], raw_shape[1]], interp = 'bilinear'), misc.imresize(labels, [raw_shape[0], raw_shape[1]], interp = 'nearest')


        return self.center_crop(images, labels, 200)


    def random_translation(self, images, labels, stride_size):
        return images

    def random_sized(self, images, labels, size):
        image_size = labels.shape
        w = int(random.uniform(0.5, 2) * image_size[0])
        h = int(random.uniform(0.5, 2) * image_size[1])
        m1,l1 = self.scale(images = cv2.resize(images, dsize = (w, h), interpolation = cv2.INTER_LINEAR), labels = cv2.resize(images, dsize = (w, h), interpolation = cv2.INTER_NEAREST), size = image_size[0])
        return self.random_crop(m1,l1,size)

    def slide_crop_old(self, images, labels, crop_size, stride_rate, ignore_label):

        w, h = labels.shape
        long_size = max(h, w)

        images = np.array(images)
        labels = np.array(labels)

        if long_size > crop_size:
            stride = int(math.ceil(crop_size * stride_rate))
            h_step_num = int(math.ceil((h - crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - crop_size) / float(stride))) + 1
            img_sublist, mask_sublist = [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + crop_size, sx + crop_size
                    img_sub = images[sy: ey, sx: ex, :]
                    mask_sub = labels[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub,crop_size)
                    img_sublist.append(img_sub)
                    mask_sublist.append(mask_sub)
            return img_sublist, mask_sublist
        else:
            images, labels = self._pad(images, labels,crop_size)
            return [images], [labels]


    def slide_crop(self, images, labels, crop_size, stride_rate, ignore_label):
        # assert images.shape == labels.shape
        w, h = labels.shape
        long_size = max(h, w)

        images = np.array(images)
        labels = np.array(labels)

        if long_size > crop_size:
            stride = int(math.ceil(crop_size * stride_rate))
            h_step_num = int(math.ceil((h - crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - crop_size) / float(stride))) + 1
            img_slices, mask_slices= [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + crop_size, sx + crop_size
                    img_sub = images[sy: ey, sx: ex, :]
                    mask_sub = labels[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub,crop_size)
                    img_slices.append(img_sub.astype(np.uint8))
                    mask_slices.append(mask_sub.astype(np.uint8))
            return img_slices, mask_slices
        else:
            images, labels= self._pad(images, labels,crop_size)
            return [images], [labels]
# , random_crop_num = 1, center_crop_num = 1, slide_crop_num = 1, slide_crop_old_num = 1

    def augmentation_final(self, raw_images = True, horizontal_flip_num = True, vertical_flip_num = True, random_rotate_num = 1, random_crop_num = 1,
                                center_crop_num = 1, slide_crop_num = 1, slide_crop_old_num = 1):
        """
        all images and labels images saved in the list are arrays format, images: w*h*3 labels: w*h*1
        [images1 images2...]  [labels1 labels2...]
        """
        images_list = []
        labels_list = []
        # n,w,h,c = self.images.shape
        n = len(self.images)
        if raw_images == True:
            for j in range(n):
                images_list.append(self.images[j])
                labels_list.append(self.labels[j])

        if horizontal_flip_num == True:
            for j in range(n):
                print('horizontal_flip')
                images_h = self.images[j]
                labels_h = self.labels[j][:,:,0]
                new_images_h,new_labels_h = self.random_horizontally_flip(images = images_h, labels = labels_h, randomly = False)
                images_list.append(new_images_h)
                labels_list.append(np.resize(new_labels_h, [self.configs['image_size'], self.configs['image_size'],1]))

        if vertical_flip_num == True:
            for j in range(n):
                print('vertical_flip')
                images_v = self.images[j]
                labels_v = self.labels[j][:,:,0]
                new_images_v,new_labels_v = self.random_vertically_flip(images = images_v, labels = labels_v, randomly = False)
                images_list.append(new_images_v)
                labels_list.append(np.resize(new_labels_v, [self.configs['image_size'], self.configs['image_size'],1]))


        if random_rotate_num > 0:
            for i in range(random_rotate_num):
                for j in range(n):
                    print('random_rotate')
                    images_r = self.images[j]
                    labels_r = self.labels[j][:,:,0]
                    new_images_r, new_labels_r = self.random_rotate(images = images_r, labels = labels_r, degree = 180)
                    images_list.append(new_images_r)
                    labels_list.append(np.resize(new_labels_r, [self.configs['image_size'], self.configs['image_size'],1]))
                    # plt.imshow(new_labels_r)
                    # plt.pause(10)

        if random_crop_num > 0:
            for i in range(random_crop_num):
                for j in range(n):
                    print('random_crop')
                    images_rc = self.images[j]
                    labels_rc = self.labels[j][:,:,0]
                    new_images_rc, new_labels_rc = self.random_crop(images = images_rc, labels = labels_rc, size = 200)
                    new_images_rc, new_labels_rc = self.scale(images = new_images_rc, labels = new_labels_rc, size = self.configs['image_size'])
                    images_list.append(new_images_rc)
                    # labels_list.append(new_labels_rc)
                    labels_list.append(np.resize(new_labels_rc, [self.configs['image_size'], self.configs['image_size'],1]))

        if center_crop_num > 0:
            for i in range(center_crop_num):
                for j in range(n):
                    print('center_crop')
                    images_cc = self.images[j]
                    labels_cc = self.labels[j][:,:,0]
                    new_images_cc, new_labels_cc = self.center_crop(images = images_cc, labels = labels_cc, size = 200)
                    new_images_cc, new_labels_cc = self.scale(images = new_images_cc, labels = new_labels_cc, size = self.configs['image_size'])
                    images_list.append(new_images_cc)
                    # labels_list.append(new_labels_cc)
                    labels_list.append(np.resize(new_labels_cc, [self.configs['image_size'], self.configs['image_size'],1]))

        if slide_crop_num > 0:
            """slide crop images returns data list contains images

            """
            for i in range(slide_crop_num):
                for j in range(n):
                    print('slide_crop')
                    images_sc = self.images[j]
                    labels_sc = self.labels[j][:,:,0]
                    new_images_sc, new_labels_sc = self.slide_crop(images = images_sc, labels = labels_sc, crop_size = 150, stride_rate = 0.2, ignore_label = True)
                    for k in range(len(new_images_sc)):
                        new_images_sc_k, new_labels_sc_k = self.scale(images = new_images_sc[k], labels = new_labels_sc[k], size = 224)
                        images_list.append(np.array(np.resize(new_images_sc_k, [self.configs['image_size'], self.configs['image_size'],3])))
                        labels_list.append(np.array(np.resize(new_labels_sc_k, [self.configs['image_size'], self.configs['image_size'],1])))



        if slide_crop_old_num > 0:
            for i in range(slide_crop_old_num):
                for j in range(n):
                    print('slide_crop_old')
                    images_sco = self.images[j]
                    labels_sco = self.labels[j][:,:,0]
                    new_images_sco, new_labels_sco = self.slide_crop_old(images = images_sco, labels = labels_sco, crop_size = 200, stride_rate = 0.5, ignore_label = True)
                    for k in range(len(new_labels_sco)):
                        new_images_sco_k, new_labels_sco_k = self.scale(images = new_images_sco[k], labels = new_labels_sco[k], size = self.configs['image_size'])
                        images_list.append(np.array(np.resize(new_images_sco_k, [self.configs['image_size'], self.configs['image_size'],3])))
                        labels_list.append(np.array(np.resize(new_labels_sco_k, [self.configs['image_size'], self.configs['image_size'],1])))

        print(len(images_list))
        return images_list,labels_list
