import cv2
import numpy as np
import os

path = "/devdata/90140_dataset/1432_first/segmentation/train/1/"

img_list = os.listdir(path)

idx = 100

for img_name in img_list:
    if img_name.endswith(".jpg"):
        print(img_name)
        path_name = path + img_name

        if img_name.endswith(".jpg"):
            img = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
            name_str1 = "./output/" + \
                img_name.split('.jpg')[0] + "_" + str(idx) + ".jpg"
            # name_str2="./ok_1/"+img_name.split('.jpg')[0]+"_"+str(idx)+"_label.jpg"

            cv2.imwrite(name_str1, img)
            # mp=np.zeros_like(img)
            # cv2.imwrite(name_str2,mp)

            label_path = path + img_name.split(".")[0] + "_1.png"
            img = (
                cv2.imread(
                    label_path,
                    cv2.IMREAD_GRAYSCALE) *
                255).astype(
                np.uint8)
            img = np.where(img > 20, 255, 0).astype(np.uint8)
            name_str2 = "./output/" + \
                img_name.split('.jpg')[0] + "_" + str(idx) + "_label.bmp"
            cv2.imwrite(name_str2, img)

        else:
            img = (
                cv2.imread(
                    path_name,
                    cv2.IMREAD_GRAYSCALE) *
                255).astype(
                np.uint8)
            img = np.where(img > 20, 255, 0).astype(np.uint8)
            name_str = "./output/" + \
                img_name.split('_1.png')[0] + "_" + str(idx) + "_label.bmp"
            cv2.imwrite(name_str, img)

        idx += 1
