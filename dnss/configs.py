"""
this .py file contains all the parameters
"""
import os
configs = {}
"""
config['model_path']
this is your model path if you have a pretrained network you need assign
this config to your directory. when you test your network you should assigh 
your directory to this too! if your trainining process is break by someone you can also 
continue your trianining process by assigh your directory to this directory

"""

#****************************************read data parameters**************************************
configs['dir_name'] = 'frame_vessel/dnss'     #need to change this dir if you change your model to a new file
configs['max_angle'] = 20
configs['root_dir'] = 'data'
configs['save_dir'] = 'saved_data'
configs['image_size'] = 224
configs['per'] = 0.9            #percentage splited from the raw data
configs['saved_npy'] = True
configs['imgs_train'] = 'imgs_train.npy'
configs['imgs_label'] = 'imgs_label.npy'
configs['imgs_train_test'] = 'imgs_train_test.npy'
configs['imgs_label_test'] = 'imgs_label_test.npy'
configs['model_path'] = "frame_vessel/pretrain_model/vgg/vgg16.npy"
# configs['model_path'] = configs['dir_name'] + '/check_points/46.npy'      
#**************************************argumentation parameters************************************
configs['raw_images'] = True
configs['horizontal_flip_num'] = True
configs['vertical_flip_num'] = True
configs['random_rotate_num'] = 0
configs['random_crop_num'] = 0
configs['center_crop_num'] = 0
configs['slide_crop_num'] = 0
configs['slide_crop_old_num'] = 0
#*************************************train parameters**********************************************
configs['image_size'] = 224
# configs['channel'] = 3
configs['channel'] = 3
configs["batch_size"] = 16
configs['epoch'] = 200
configs['final_layer_type'] = "softmax_sparse"
configs['learning_rate_orig'] = 1e-3
configs['checkpoint_dir'] = configs['dir_name']+ '/check_points'
configs['num_classes'] = 3
configs['VGG_MEAN'] = [103.94,116.78,123.68]
configs['_BATCH_NORM_DECAY'] = 0.997
configs['_BATCH_NORM_EPSILON'] = 1e-5
configs['save_frequency'] = 100
configs['trained_parameters'] = configs['checkpoint_dir'] + '/'
#************************************device parameters**********************************************
configs["num_gpus"] = 1
configs["dev"] = '/gpu:1'  #'/cpu:0'
configs['GPU'] = '2'	
# configs["dev"] = '/cpu:0'  #'/cpu:0'
configs['tensorboard_dir'] = configs['dir_name'] + '/Tfboard/Result/'
configs['tensorboard_on'] = True
configs['tensorboard_refresh'] = 50
#************************************evaluate parameters********************************************
configs['test_num'] = 20

