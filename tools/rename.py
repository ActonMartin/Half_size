import cv2
import numpy as np
import os

path = "/devdata/90140_dataset/1432_first/segmentation/train/1/"

img_list = os.listdir(path)


for img_name in img_list:
    print(img_name)
    if img_name.endswith("label.jpg"):
        path_name = path + img_name
        print(img_name)
        img = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
        name_str1 = path_name.split(".")[0] + ".bmp"
        print("=====", name_str1)
        cv2.imwrite(name_str1, img)
        os.remove(path_name)
