import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import embedding_ops
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import math_ops


class Model(object):
    def __init__(self, sess, param):
        self.step = 0
        self.__session = sess
        self.is_training = True
        self.__learn_rate = param["learn_rate"]
        self.__learn_rate = param["learn_rate"]
        self.__max_to_keep = param["max_to_keep"]
        self.__checkPoint_dir = param["checkPoint_dir"]
        self.__restore = param["b_restore"]
        self.__mode = param["mode"]
        self.input_size = param["input_size"]
        self.num_classes = param["num_classes"]

        self.is_training = True
        self.__batch_size = param["batch_size"]
        if self.__mode is "savaPb":
            self.__batch_size = 1

        # Building graph
        with self.__session.as_default():
            self.build_model()

        # 参数初始化，或者读入参数
        with self.__session.as_default():
            self.init_op.run()
            self.__saver = tf.train.Saver(
                tf.global_variables(),
                max_to_keep=self.__max_to_keep)
            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if ckpt:
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
                    print('Restoring from epoch:{}'.format(self.step))
                    self.step += 1

    def build_model(self):

        def conv_concat(inputA, input_B, name):

            return tf.concat([inputA, input_B], axis=-1,
                             name="concat_{}".format(name))

        def upconv_concat(inputA, input_B, n_filter, flags, name):
            """Upsample `inputA` and concat with `input_B`
            Args:
                input_A (4-D Tensor): (N, H, W, C)
                input_B (4-D Tensor): (N, 2*H, 2*H, C2)
                name (str): name of the concat operation
            Returns:
                output (4-D Tensor): (N, 2*H, 2*W, C + C2)
            """
            up_conv = upconv_2D(inputA, n_filter, flags, name)

            return tf.concat([up_conv, input_B], axis=-1,
                             name="concat_{}".format(name))

        def upconv_2D(tensor, n_filter, flags, name):
            """Up Convolution `tensor` by 2 times
            Args:
                tensor (4-D Tensor): (N, H, W, C)
                n_filter (int): Filter Size
                name (str): name of upsampling operations
            Returns:
                output (4-D Tensor): (N, 2 * H, 2 * W, C)
            """
            reg = 0.1
            return tf.layers.conv2d_transpose(
                tensor,
                filters=n_filter,
                kernel_size=2,
                strides=2,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                name="upsample_{}".format(name))

        def SegmentNet(input, scope, is_training, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm):

                    net = slim.conv2d(input, 16, [3, 3], scope='conv1')
                    net = slim.conv2d(net, 16, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool1')

                    net = slim.conv2d(net, 32, [3, 3], scope='conv3')
                    net = slim.conv2d(net, 32, [3, 3], scope='conv4')
                    #net = slim.conv2d(net, 32, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool2')

                    net = slim.conv2d(net, 32, [3, 3], scope='conv6')
                    #net = slim.conv2d(net, 32, [3, 3], scope='conv7')
                    net = slim.conv2d(net, 32, [3, 3], scope='conv8')
                    net = slim.conv2d(net, 32, [3, 3], scope='conv9')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool3')

                    net1 = slim.conv2d(net, 128, [1, 1], scope='conv10')
                    net2 = tf.keras.layers.Conv2D(
                        128, 3, strides=1, padding="same", dilation_rate=3)(net)
                    m1 = conv_concat(net1, net2, 1)
                    net2 = tf.keras.layers.Conv2D(
                        128, 3, strides=1, padding="same", dilation_rate=6)(net)
                    m2 = conv_concat(m1, net2, 2)
                    net2 = tf.keras.layers.Conv2D(
                        128, 3, strides=1, padding="same", dilation_rate=12)(net)
                    net = conv_concat(m2, net2, 3)

                    net = slim.conv2d(net, 512, [1, 1], scope='conv11')

                    features = net

                    net = slim.conv2d(
                        net, 1, [
                            1, 1], activation_fn=None, scope='conv17')
                    logits_pixel = net
                    net = tf.sigmoid(net, name=None)
                    mask = net

            return features, logits_pixel, mask

        def DecisionNet(
                feature,
                logits_pixel,
                mask,
                scope,
                is_training,
                num_classes=2,
                reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                with slim.arg_scope([slim.conv2d],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm):

                    net = tf.concat([feature, mask], axis=3)
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool1')
                    net = slim.conv2d(net, 8, [5, 5], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool2')
                    net = slim.conv2d(net, 16, [5, 5], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool3')
                    net = slim.conv2d(net, 32, [5, 5], scope='conv3')
                    vector1 = math_ops.reduce_mean(
                        net, [1, 2], name='pool4', keepdims=True)
                    vector2 = math_ops.reduce_max(
                        net, [1, 2], name='pool5', keepdims=True)
                    vector3 = math_ops.reduce_mean(
                        mask, [1, 2], name='pool6', keepdims=True)
                    vector4 = math_ops.reduce_max(
                        mask, [1, 2], name='pool7', keepdims=True)
                    vector = tf.concat(
                        [vector1, vector2, vector3, vector4], axis=3)
                    vector = tf.squeeze(vector, axis=[1, 2])
                    logits = slim.fully_connected(
                        vector, num_classes, activation_fn=None)
                    logits = tf.sigmoid(logits, name=None)
                    # output=tf.argmax(logits,axis=1)
                    output = logits
                    return logits, output

        Image = tf.placeholder(
            tf.float32,
            shape=(
                self.__batch_size,
                self.input_size[0],
                self.input_size[1],
                1),
            name='Image')
        PixelLabel = tf.placeholder(
            tf.float32,
            shape=(
                self.__batch_size,
                self.input_size[0] / 8,
                self.input_size[1] / 8,
                1),
            name='PixelLabel')
        Label_class = tf.placeholder(
            tf.int32,
            shape=(
                self.__batch_size),
            name='Label')  # 原先是：tf.int32
        features, logits_pixel, mask = SegmentNet(
            Image, 'segment', self.is_training)
        logits_class, output_class = DecisionNet(
            features, logits_pixel, mask, 'decision', self.is_training, self.num_classes)
        #logits_class=DecisionNet(features, logits_pixel, feat1, feat2, mask, 'decision', self.is_training, self.num_classes)
        # 损失函数
        logits_pixel = tf.reshape(logits_pixel, [self.__batch_size, -1])
        PixelLabel_reshape = tf.reshape(PixelLabel, [self.__batch_size, -1])
        loss_pixel = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_pixel,
                labels=PixelLabel_reshape))
        # loss_class =
        # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_class,labels=Label_class))##decision的损失
        loss_class = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_class, labels=Label_class))

        loss_total = loss_pixel + loss_class
        optimizer = tf.train.GradientDescentOptimizer(self.__learn_rate)
        train_var_list = [v for v in tf.trainable_variables()]
        train_segment_var_list = [
            v for v in tf.trainable_variables() if 'segment' in v.name]
        train_decision_var_list = [
            v for v in tf.trainable_variables() if 'decision' in v.name]
        optimize_segment = optimizer.minimize(
            loss_pixel, var_list=train_segment_var_list)
        optimize_decision = optimizer.minimize(
            loss_class, var_list=train_decision_var_list)  # decision的优化器
        optimize_total = optimizer.minimize(
            loss_total, var_list=train_var_list)
        init_op = tf.global_variables_initializer()
        self.Image = Image
        self.PixelLabel = PixelLabel
        self.Label_class = Label_class
        self.features = features
        self.mask = mask
        self.logits_class = logits_class
        self.output_class = output_class
        self.loss_pixel = loss_pixel
        self.loss_class = loss_class  # decision的损失
        self.loss_total = loss_total
        self.optimize_segment = optimize_segment
        self.optimize_decision = optimize_decision  # decision的优化器
        self.optimize_total = optimize_total
        self.init_op = init_op

    def save(self):
        self.__saver.save(
            self.__session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
        )
