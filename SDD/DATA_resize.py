import os
from PIL import Image
"""
这个代码的作用是resize
"""

folder_path = "E:/DATA/3_fold - 副本/"

for root, dirs, files in os.walk(folder_path):
	for file in files:
		if file.endswith('.jpg') or file.endswith('.bmp'):
			image_path = os.path.join(root, file)
			img = Image.open(image_path)
			(width, height) = img.size
			new_size = (int(width/2), int(height/2))
			new_image = img.resize(new_size)
			new_image.save(image_path)
			print(image_path)
