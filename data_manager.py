import re
import os
import numpy as np
import cv2
import tensorflow as tf

from data_augment import load_image, get_seq, aug_data
from scipy.misc import imread, imresize, imsave
from random import shuffle


class DataManager(object):
    def __init__(self, dataList, param, shuffle=True, is_positive=True):
        self.shuffle = shuffle
        self.is_positive = is_positive
        self.data_list = dataList
        self.data_size = len(dataList)
        self.data_dir = param["data_dir"]
        self.epochs_num = param["epochs_num"]
        self.batch_size = param["batch_size"]
        self.train_mode = param['train_mode']
        self.num_classes = param['num_classes']
        self.input_size = param['input_size']
        self.number_batch = int(
            np.floor(len(self.data_list) / self.batch_size))
        self.next_batch = self.get_next()
        self.seq = get_seq()

    def get_next(self):
        if self.train_mode == 'segment':
            dataset = tf.data.Dataset.from_generator(
                self.generator,
                (tf.float32,
                 tf.float32,
                 tf.int32,
                 tf.string))  # 原先second is tf.int32
        else:
            dataset = tf.data.Dataset.from_generator(
                self.generator,
                (tf.float32,
                 tf.float32,
                 tf.int32,
                 tf.string))  # 原先second is tf.int32

        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(
                self.batch_size * 3 + 300)  # 将数据打乱，数值越大，混乱程度越大

        # 按照顺序取出self.batch_size行数据，最后一次输出可能小于batch
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch
    '''
    def generator(self):
        for index in range(len(self.data_list)):
            file_basename_image,file_basename_label = self.data_list[index]
            image_path = os.path.join(self.data_dir, file_basename_image)
            label_path= os.path.join(self.data_dir, file_basename_label)
            image= self.read_data(image_path, is_label=False)

            if self.is_positive==False:
                 label = np.zeros((self.input_size[0], self.input_size[1]))
            else:
                 label = self.read_data(label_path, is_label=True)

            label_pixel, label = self.label_preprocess(label)
            image = (np.array(image[:, :, np.newaxis]))

            if not self.train_mode=='decision': ###decision模式下，维度要减少一个，所以不用np.newaxis进行扩充，这是由sparse_softmat_cross_entropy函数决定的
                  label_pixel = (np.array(label_pixel[:, :, np.newaxis]))

            image=image.astype(np.float32)
            if self.train_mode=='segment':
                label_pixel=label_pixel.astype(np.float32)
            else:
                label_pixel=label_pixel.astype(np.int32)
            label=label

            yield image, label_pixel, label, file_basename_image
    '''
    '''
    def read_data(self, data_name, is_label):
        img = cv2.imread(data_name, 0)  # /255.#read the gray image
        if img.shape[0]>img.shape[1]:
            img= cv2.flip(img, 1)  #原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
            img = cv2.transpose(img)

        img = cv2.resize(img, (self.input_size[1], self.input_size[0]), cv2.INTER_NEAREST)  ##resize(img, (width, height))

        # img = img.swapaxes(0, 1)
        # image = (np.array(img[:, :, np.newaxis]))
        return img
    '''

    def generator(self):
        for index in range(len(self.data_list)):
            file_basename_image, file_basename_label = self.data_list[index]
            image_path = os.path.join(self.data_dir, file_basename_image)
            label_path = os.path.join(self.data_dir, file_basename_label)
            image = load_image(image_path, False, self.input_size)
            if not self.is_positive:
                label = np.zeros(
                    (self.input_size[0], self.input_size[1]))[
                    np.newaxis, :, :, np.newaxis]
                label = label.astype(np.float32)
                image, label = aug_data(self.seq, image, label)
                image = image.squeeze(3)
                image = image.squeeze(0)
                label = label.squeeze(3)
                label = label.squeeze(0)
                image = cv2.resize(
                    image, (self.input_size[1], self.input_size[0]))
                label = cv2.resize(
                    label, (self.input_size[1], self.input_size[0]))
            else:
                label = load_image(label_path, True, self.input_size)
                image, label = aug_data(self.seq, image, label)
                image = image.squeeze(3)
                image = image.squeeze(0)
                label = label.squeeze(3)
                label = label.squeeze(0)
                label = self.proc(label, label_path, is_label=True)

            label_pixel, label = self.label_preprocess(label)
            image = (np.array(image[:, :, np.newaxis]))

            if not self.train_mode == 'decision':
                # decision模式下，维度要减少一个，所以不用np.newaxis进行扩充，这是由sparse_softmat_cross_entropy函数决定的
                label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
            image = image.astype(np.float32)
            if self.train_mode == 'segment':
                label_pixel = label_pixel.astype(np.float32)
            else:
                label_pixel = label_pixel.astype(np.int32)
            yield image, label_pixel, label, file_basename_image

    def proc(self, img, data_name, is_label):
        """
        if is_label and self.train_mode=='decision':
             class_id=int(data_name.split("/")[-2].split(",")[0])
             if not class_id in range(self.num_classes):
                  raise ValueError("class_id speficied by folder name is wrong, class_id = ", class_id)
             img=(img>0).astype(np.uint8)*class_id ##乘以一个类别，就是分割时候语义的类别
        """
        img = cv2.resize(
            img,
            (self.input_size[1],
             self.input_size[0]),
            cv2.INTER_LINEAR)

        # img = img.swapaxes(0, 1)
        # image = (np.array(img[:, :, np.newaxis]))
        return img

    def label_preprocess(self, label):
        if not self.train_mode == 'decision':
            label_ = cv2.resize(label,
                                (int(self.input_size[1] / 8),
                                 int(self.input_size[0] / 8)),
                                cv2.INTER_LINEAR)
            label_ = self.ImageBinarization(label_)
        else:
            label_ = cv2.resize(label,
                                (int(self.input_size[1] / 8),
                                 int(self.input_size[0] / 8)),
                                cv2.INTER_NEAREST)
        label = label_.sum()
        if label > 0:
            label = 1
        return label_, label

    def ImageBinarization(self, img, threshold=0):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image
