import os
'''
代码的作用是删除文件里面的7Z压缩包
'''

folder_path = "E:/DATA/3_fold_half_pix/"

for root, dirs, files in os.walk(folder_path):
	for file in files:
		if file.endswith('.7z') or file.endswith('zip'):
			compress_path = os.path.join(root, file)
			os.remove(compress_path)