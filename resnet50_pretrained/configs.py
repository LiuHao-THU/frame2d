"""
this .py file contains all the parameters
"""
import os
configs = {}
main_dir = 'frame_vessel/resnet50_pretrained'
#****************************************read data parameters**************************************
configs['max_angle'] = 20
configs['root_dir'] = 'data'
configs['save_dir'] = 'saved_data'
configs['image_size'] = 224
configs['per'] = 0.9                      #percentage splited from the raw data
configs['saved_npy'] = True
configs['imgs_train'] = 'imgs_train.npy'
configs['imgs_label'] = 'imgs_label.npy'
configs['imgs_train_test'] = 'imgs_train_test.npy'
configs['imgs_label_test'] = 'imgs_label_test.npy'
configs['model_path'] = 'frame_vessel/pretrain_model/resnet/resnet50.npy'
#**************************************argumentation parameters************************************
configs['raw_images'] = True
configs['horizontal_flip_num'] = False
configs['vertical_flip_num'] = False
configs['random_rotate_num'] = 1
configs['random_crop_num'] = 1
configs['center_crop_num'] = 0
configs['slide_crop_num'] = 0
configs['slide_crop_old_num'] = 0
#*************************************train parameters**********************************************
configs['image_size'] = 224
# configs['channel'] = 3
configs['channel'] = 3
configs["batch_size"] = 32
configs['epoch'] = 20
configs['final_layer_type'] = "softmax_sparse"
configs['learning_rate_orig'] = 1e-3
configs['checkpoint_dir'] = main_dir+ '/check_points'
configs['num_classes'] = 3
configs['VGG_MEAN'] = [1.030626238009759419e+02, 1.159028825738600261e+02, 1.231516308384586438e+02]
configs['_BATCH_NORM_DECAY'] = 0.997
configs['_BATCH_NORM_EPSILON'] = 1e-5
#************************************device parameters**********************************************
configs["num_gpus"] = 1
configs["dev"] = '/gpu:0'  #'/cpu:0'
# configs["dev"] = '/cpu:0'  #'/cpu:0'
configs['GPU'] = '0'

#************************************evaluate parameters********************************************


