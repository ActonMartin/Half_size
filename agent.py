import tensorflow as tf
import numpy as np
import shutil
import os
import cv2
import random
import utils

from data_manager import DataManager
from model import Model
from datetime import datetime


class Agent(object):
    def __init__(self, param):
        self.__Param = param
        self.gpu_config()
        self.init_datasets()  # 初始化数据管理器
        self.model = Model(self.__sess, self.__Param)  # 建立模型
        self.logger = utils.get_logger(param["Log_dir"])

    def gpu_config(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.__Param['gpu']  # 指定第一块GPU可用
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 程序最多只能占用指定gpu 95%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        self.__sess = tf.Session(config=config)

    def run(self):
        if self.__Param["mode"] is "training":
            train_mode = self.__Param["train_mode"]
            self.train(train_mode)
        elif self.__Param["mode"] is "testing":
            path_which_folder_test = self.__Param["path_which_folder_test"]
            self.test(path_which_folder_test)
        elif self.__Param["mode"] is "savePb":
            raise Exception(" this  mode is incomplete ")
        else:
            print(
                "got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

    def init_datasets(self):
        self.Positive_data_list, self.Negative_data_list = self.listData(
            self.__Param["data_dir"])
        if self.__Param["mode"] is "training":
            self.DataManager_train_Positive = DataManager(
                self.Positive_data_list, self.__Param, shuffle=True, is_positive=True)
            self.DataManager_train_Negative = DataManager(
                self.Negative_data_list, self.__Param, shuffle=True, is_positive=False)
        elif self.__Param["mode"] is "testing":
            self.DataManager_test_Positive = DataManager(
                self.Positive_data_list, self.__Param, shuffle=False, is_positive=True)
            self.DataManager_test_Negative = DataManager(
                self.Negative_data_list, self.__Param, shuffle=False, is_positive=False)
        elif self.__Param["mode"] is "savePb":
            pass
        else:
            raise Exception('got a unexpected  mode ')

    def train(self, mode):
        if mode not in ["segment", "decision", "total"]:
            raise Exception(
                'got a unexpected  training mode ,options :{segment,decision}')

        with self.__sess.as_default():
            self.logger.info('start training {} net'.format(mode))
            for i in range(
                    self.model.step,
                    self.__Param["epochs_num"] +
                    self.model.step):
                # epoch start
                print("......at epoch: ", i)
                iter_loss = 0
                for batch in range(
                        self.DataManager_train_Positive.number_batch):
                    # batch start
                    for index in range(2):
                        # corss training the positive sample and negative
                        # sample
                        if index == 0:
                            img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(
                                self.DataManager_train_Positive.next_batch)
                        else:
                            img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(
                                self.DataManager_train_Negative.next_batch)
                        loss_value_batch = 0
                        #print("dtype---------------------------", img_batch[0].dtype,label_pixel_batch[0].dtype )
                        if mode == "segment":
                            _, loss_value_batch = self.__sess.run([self.model.optimize_segment, self.model.loss_pixel],
                                                                  feed_dict={self.model.Image: img_batch,
                                                                             self.model.PixelLabel: label_pixel_batch})
                        elif mode == "decision":
                            _, loss_value_batch = self.__sess.run([self.model.optimize_decision, self.model.loss_class],
                                                                  feed_dict={self.model.Image: img_batch,
                                                                             self.model.Label_class: label_batch})
                        '''
                        elif mode == "total":
                            _, loss_value_batch = self.__sess.run([self.model.optimize_total, self.model.loss_total],
                                                                  feed_dict={self.model.Image: img_batch,
                                                                             self.model.PixelLabel: label_pixel_batch,
                                                                             self.model.Label: label_batch})
                        '''
                        iter_loss += loss_value_batch
                        # 可视化
                        if i % self.__Param["valid_frequency"] == 0 and i > 0:
                            if mode == "decision":
                                output_batch = self.__sess.run(
                                    self.model.output_class, feed_dict={
                                        self.model.Image: img_batch})
                            else:
                                output_batch = self.__sess.run(self.model.mask, feed_dict={
                                                               self.model.Image: img_batch})
                                save_dir = "./visualization/train/training_epoch-{}".format(i)
                                self.visualization(
                                    img_batch, output_batch, file_name_batch, save_dir)
                self.logger.info(
                    'epoch:[{}] ,train_mode:{}, loss: {}'.format(
                        self.model.step, mode, iter_loss))
                # 保存模型
                if i % self.__Param["save_frequency"] == 0 or i == self.__Param["epochs_num"] + \
                        self.model.step - 1:
                    self.model.save()
                # #验证
                # if i % self.__Param["valid_frequency"] == 0 and i>0:
                # 	self.valid()
                self.model.step += 1
    '''
    def test(self):
        # anew a floder to save visualization
        visualization_dir = "./visualization/test/"
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        with self.__sess.as_default():
            self.logger.info('start testing')
            DataManager = [self.DataManager_test_Positive, self.DataManager_test_Negative]
            for index in range(2):
                for batch in range(DataManager[index].number_batch):
                    img_batch, label_pixel_batch, file_name_batch, = self.__sess.run(
                        DataManager[index].next_batch)

                    mask_batch, output_batch = self.__sess.run([self.model.mask, self.model.output_class],
                                                               feed_dict={self.model.Image: img_batch, })
                    self.visualization(img_batch, output_batch, mask_batch, file_name_batch,
                                       save_dir=visualization_dir)
    '''

    def test(self, path_which_folder_test):
        # anew a floder to save visualization
        #path_which_folder_test = "E:/DATA/3_fold_half_pix/fold_0/dilate_0/" # 测试的时候修改这个位置
        path_test_set = path_which_folder_test + "test_set/"
        visualization_dir = path_which_folder_test + "learn_rate_" + str(self.__Param["learn_rate"]) + \
            self.__Param["train_test_visualization"]  # 测试的时候修改这个位置
        #    "/train1test5/visualization5/"  # 测试的时候修改这个位置
        visualization_dir_ok = visualization_dir + "ok/"
        visualization_dir_ng = visualization_dir + "ng/"
        if not os.path.exists(visualization_dir_ok):
            os.makedirs(visualization_dir_ok)
        if not os.path.exists(visualization_dir_ng):
            os.makedirs(visualization_dir_ng)
        with self.__sess.as_default():
            self.logger.info('start testing')
            #path = "E:/DATA/3_fold/fold_0/dilate_0/test_set/"
            # path='/devdata/90140/thick7exp/train/sides/lr_ends_prob/dataset/segment/WuZi,positive/L1,positive/'
            file_list = os.listdir(path_test_set)
            num = 0
            for f in file_list:
                if f.endswith(".bmp") or os.path.isdir(path_test_set + f):
                    continue
                img_batch = cv2.imread(path_test_set + f, 0)
                img_batch = cv2.resize(
                    img_batch,
                    (self.__Param["input_size"][1],
                     self.__Param["input_size"][0]),
                    cv2.INTER_NEAREST)
                '''
                if img_batch.shape[0]>img_batch.shape[1]:
                   img_batch = cv2.flip(img_batch, 1)  #原型：cv2.flip(src, flipCode[, dst]) → dst  flipCode表示对称轴 0：x轴  1：y轴.  -1：both
                   img_batch = cv2.transpose(img_batch)
                '''
                img_batch = img_batch[np.newaxis, :, :, np.newaxis]

                mask_batch, output_batch = self.__sess.run([self.model.mask, self.model.output_class],
                                                           feed_dict={self.model.Image: img_batch, })
                # self.visualization(img_batch, output_batch, mask_batch, file_name_batch,
                #                   save_dir=visualization_dir)
                res="ok" if output_batch[0][0]>output_batch[0][1] else "ng"
                #res="ok" if output_batch[0][0]>output_batch[0][1] and output_batch[0][0]>0.8 else "ng"
                #res = "ng" if (
                #    output_batch[0][0] < output_batch[0][1] and output_batch[0][1] > 0.9) else "ok"
                # if res=="ok":
                print(
                    "/////...........",
                    res,
                    '=====',
                    f,
                    output_batch[0][0],
                    output_batch[0][1],
                    num)
                num += 1
                if res == 'ok':
                    out_path_img = visualization_dir_ok + f + "_img.jpg"
                    out_path_msk = visualization_dir_ok + f + "_msk.jpg"
                else:
                    out_path_img = visualization_dir_ng + f + "_img.jpg"
                    out_path_msk = visualization_dir_ng + f + "_msk.jpg"
                cv2.imwrite(out_path_img, img_batch.squeeze())
                cv2.imwrite(out_path_msk, mask_batch.squeeze() * 255)
                #print("/////...........", output_batch ,'=====', f)

                #cv2.imshow("xxx", img_batch.squeeze())
                # mask_batch=mask_batch*255
                #bt=cv2.resize(mask_batch.squeeze().astype(np.uint8), (img_batch.shape[2], img_batch.shape[1]))
                #cv2.imshow("mask", bt)
                # cv2.waitKey(0)

    def valid(self):
        pass

    def visualization(
            self,
            img_batch,
            output_batch,
            filenames,
            save_dir="./new_visualization"):
        # anew a floder to save visualization
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, filename in enumerate(filenames):
            filename = os.path.basename(filename)
            mask = np.array(output_batch[i]).squeeze(2)
            image = np.array(img_batch[i]).squeeze(2)
            mask = mask * 255
            mask = cv2.resize(
                mask.astype(
                    np.uint8), (image.shape[1], image.shape[0]))
            filename = str(filename).split("'")[-2]  # 原本像这样： b'a1_513_6_B.jpg'
            path_1 = os.path.join(
                save_dir,
                filename.split(".")[0] +
                filename.split(".")[1] +
                "_img.jpg")
            path_2 = os.path.join(
                save_dir,
                filename.split(".")[0] +
                filename.split(".")[1] +
                "_mask.jpg")
            cv2.imwrite(path_1, image)
            cv2.imwrite(path_2, mask)

    def listData(self, data_dir, valid_ratio=0.05):
        def judge_positive_label(img_full_path):  # 判断标注的图片里面是否真的有标注信息
            label_map_path = img_full_path.split(
                '.')[0] + self.__Param['label_subfix']  # '_label.bmp'
            img = cv2.imread(label_map_path, cv2.IMREAD_GRAYSCALE)
            if img.sum() > 0:
                return True, label_map_path
            else:
                return False, label_map_path

        Positive_examples_list = []
        Negative_examples_list = []
        Positive_examples_train = []
        Negative_examples_train = []
        Positive_examples_valid = []
        Negative_examples_valid = []

        for root, dirs, files in os.walk(data_dir):
            for name in files:
                # 文件夹的名称，通过文件夹的名称即可判断是positive, negative 还是ignore样本
                file_name = root.split("/")[-1]
                img_full_path = os.path.join(root, name)
                if name.endswith(".jpg"):
                    token = file_name.split(",")[-1]
                    if token == 'positive':
                        is_positive, label_map_path = judge_positive_label(
                            img_full_path)
                        if is_positive:
                            Positive_examples_list.append(
                                [img_full_path, label_map_path])
                        else:
                            Negative_examples_list.append(
                                [img_full_path, label_map_path])
                    elif token == 'negative':
                        label_map_path = img_full_path.split(
                            '.')[0] + self.__Param['label_subfix']  # '_label.bmp'
                        Negative_examples_list.append(
                            [img_full_path, label_map_path])

                    elif token == 'ignore':
                        continue

                    else:
                        raise Exception(
                            'bad tokens, it should be one of positive, negative, or ignore')

        random.shuffle(Positive_examples_list)
        random.shuffle(Negative_examples_list)

        def train_test_list(is_positive, examples_list):
            valid_offset = np.floor(len(examples_list) * valid_ratio)
            for i in range(len(examples_list)):
                if i < valid_offset:
                    if is_positive:
                        Positive_examples_valid.append(examples_list[i])
                    else:
                        Negative_examples_valid.append(examples_list[i])
                else:
                    if is_positive:
                        Positive_examples_train.append(examples_list[i])
                    else:
                        Negative_examples_train.append(examples_list[i])

        train_test_list(True, Positive_examples_list)
        train_test_list(False, Negative_examples_list)

        if self.__Param["mode"] is "training":
            '''
            for p in Positive_examples_train:
                print("-----------pos-----------", p)
            for p in Negative_examples_train:
                print("-----------neg-----------", p)
            '''
            return Positive_examples_train, Negative_examples_train

        if self.__Param["mode"] is "testing":
            return Positive_examples_train, Negative_examples_train
