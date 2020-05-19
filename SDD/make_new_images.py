import os
from PIL import Image
"""
这个代码主要是修改图片的名称

使用讀取的方式修改文件名会造成图片大小变化，就不是源文件了。
		img = Image.open(image_path)
		img.save(change_image_path)
"""

path = "E:/DATA/KolektorSDD/"
change_path = "E:/DATA/KolektorSDD_change/"
if not os.path.exists(change_path):
	os.makedirs(change_path)

all_folders = os.listdir(path)
for each_folder in all_folders:
	image_folder = path + each_folder
	change_image_folder = change_path + each_folder
	# if not os.path.exists(change_image_folder):
	# 	os.makedirs(change_image_folder)

	all_images = os.listdir(image_folder)
	for image in all_images:
		image_path = os.path.join(image_folder, image)
		new_image_name = each_folder + "_" + image
		change_image_path = os.path.join(change_path, new_image_name)
		#print(change_image_path)
		os.rename(image_path, change_image_path)
		print("成功修改{0}的名称".format(image))

