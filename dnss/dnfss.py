import tensorflow as tf
from .configs import configs
from .CNN import *
class DNFSS():
    def __init__(self, x, y, rate,image):
        self.x = x
        self.y = y
        self.image = image
        self.rate = rate

    def build(self):
        # self.x_batch_train = x_batch_train
        # self.y_batch_train = y_batch_train
        # self.x_batch_validation = x_batch_validation
        # self.y_batch_validation = y_batch_validation
        self.x_image = self.x
        self.y_ = self.y

        # self.expected = tf.expand_dims(self.y_, -1)

        self.conv_1_1 = conv_layer(self.x_image, [3, 3, 3, 64], 64, 'conv_1_1')
        self.conv_1_2 = conv_layer(self.conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')

        self.pool_1, self.pool_1_argmax = pool_layer(self.conv_1_2)

        self.conv_2_1 = conv_layer(self.pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
        self.conv_2_2 = conv_layer(self.conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

        self.pool_2, self.pool_2_argmax = pool_layer(self.conv_2_2)

        self.conv_3_1 = conv_layer(self.pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
        self.conv_3_2 = conv_layer(self.conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
        self.conv_3_3 = conv_layer(self.conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

        self.pool_3, self.pool_3_argmax = pool_layer(self.conv_3_3)

        self.conv_4_1 = conv_layer(self.pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
        self.conv_4_2 = conv_layer(self.conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
        self.conv_4_3 = conv_layer(self.conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

        self.pool_4, self.pool_4_argmax = pool_layer(self.conv_4_3)

        self.conv_5_1 = conv_layer(self.pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
        self.conv_5_2 = conv_layer(self.conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
        self.conv_5_3 = conv_layer(self.conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

        self.pool_5, self.pool_5_argmax = pool_layer(self.conv_5_3)

        self.fc_6 = conv_layer(self.pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
        self.fc_7 = conv_layer(self.fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

        self.deconv_fc_6 = deconv_layer(self.fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

        self.unpool_5 = unpool(self.deconv_fc_6, self.pool_5_argmax)

        self.deconv_5_3 = deconv_layer(self.unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        self.deconv_5_2 = deconv_layer(self.deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        self.deconv_5_1 = deconv_layer(self.deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        self.unpool_4 = unpool(self.deconv_5_1, self.pool_4_argmax)

        self.deconv_4_3 = deconv_layer(self.unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        self.deconv_4_2 = deconv_layer(self.deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        self.deconv_4_1 = deconv_layer(self.deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        self.unpool_3 = unpool(self.deconv_4_1, self.pool_3_argmax)

        self.deconv_3_3 = deconv_layer(self.unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        self.deconv_3_2 = deconv_layer(self.deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        self.deconv_3_1 = deconv_layer(self.deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        self.unpool_2 = unpool(self.deconv_3_1, self.pool_2_argmax)

        self.deconv_2_2 = deconv_layer(self.unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        self.deconv_2_1 = deconv_layer(self.deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        self.unpool_1 = unpool(self.deconv_2_1, self.pool_1_argmax)

        self.deconv_1_2 = deconv_layer(self.unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        self.deconv_1_1 = deconv_layer(self.deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

        self.y_conv = deconv_layer(self.deconv_1_1, [1, 1, 3, 32], 3, 'score_1')
        # print(self.y_conv)
        self.y_soft = tf.nn.softmax(self.y_conv)
        # self.xx = self.y_soft
        logits = tf.reshape(self.y_conv, (-1, 3))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.reshape(self.y_, [-1]), name='x_entropy')
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

        self.cross_entropy_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.rate).minimize(self.loss)

        #tf.summary.image('input x_image', self.x_image, 4)
        #tf.summary.image('y_prediction', self.y_conv, 4)
        #tf.summary.image('y_GT', self.y_, 4)
        #tf.summary.image('y_pred_softmax', self.y_soft, 4)
        tf.summary.scalar('cross_entropy', self.loss)
        # self.xe_valid_summary = tf.summary.scalar('cross_entropy_valid', self.cross_entropy_valid)
        #tf.summary.scalar('learning rate', self.lr)
