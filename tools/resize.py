import cv2
import numpy as np
import os

path = "/devdata/90140_dataset/1432_first/segmentation/train/KOS/kos25/"

img_list = os.listdir(path)

k = 4
sz = (320 * k, 320 * k)

for img_name in img_list:
    print(img_name)

    path_name = path + img_name
    print(img_name)
    img = cv2.imread(path_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, sz)
    cv2.imwrite(path_name, img)
