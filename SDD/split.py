import os
import shutil

folders = "E:/DATA/3_fold/"
sub_folders = os.listdir(folders)
ok = "E:/DATA/KolektorSDD_MOD/OK,negative/"
ng = "E:/DATA/KolektorSDD_MOD/NG,positive/"
images_ok = os.listdir(ok)
images_ng = os.listdir(ng)

for each_folder in sub_folders:
	path_sub = os.path.join(folders, each_folder)
	sub_files = os.listdir(path_sub)

	train_folder = os.path.join(path_sub,"train_set")
	test_folder = os.path.join(path_sub, "test_set")
	sub_ok_folder = os.path.join(train_folder, "OK,negative")
	sub_ng_folder = os.path.join(train_folder, "NG,positive")
	if not os.path.exists(sub_ok_folder):
		os.makedirs(sub_ok_folder)
	if not os.path.exists(sub_ng_folder):
		os.makedirs(sub_ng_folder)
	if not os.path.exists(test_folder):
		os.makedirs(test_folder)

	for p in sub_files:
		if p == "train_ids.txt":
			train_set = []
			p_path = os.path.join(path_sub, p)
			f = open(p_path)
			lines = f.readlines()
			count = 0
			for line in lines:
				s = line.strip("\n").split("/")[-1]
				train_set.append(s)
			#print(train_set)
			for each_image_ok in images_ok:
				#print(each_image_ok)
				this_ok_image_path = os.path.join(ok, each_image_ok)
				print(this_ok_image_path)
				pre_fix = each_image_ok.split("_")[0]
				#print(pre_fix)
				if pre_fix in train_set:
					target_ok_path = os.path.join(sub_ok_folder, each_image_ok)
					#print(target_path)
					shutil.copy(this_ok_image_path, target_ok_path)

			for each_image_ng in images_ng:
				#print(each_image_ok)
				this_ng_image_path = os.path.join(ng, each_image_ng)
				#print(this_ng_image_path)
				pre_fix = each_image_ng.split("_")[0]
				#print(pre_fix)
				if pre_fix in train_set:
					target_ng_path = os.path.join(sub_ng_folder, each_image_ng)
					#print(target_path)
					shutil.copy(this_ng_image_path, target_ng_path)

		if p == "test_ids.txt":
			test_set = []
			p_path = os.path.join(path_sub, p)
			f = open(p_path)
			lines = f.readlines()
			count = 0
			for line in lines:
				s = line.strip("\n").split("/")[-1]
				test_set.append(s)
			#print(train_set)
			for each_image_ok in images_ok:
				#print(each_image_ok)
				this_ok_image_path = os.path.join(ok, each_image_ok)
				print(this_ok_image_path)
				pre_fix = each_image_ok.split("_")[0]
				#print(pre_fix)
				if pre_fix in test_set:
					target_ok_path = test_folder
					#print(target_path)
					shutil.copy(this_ok_image_path, target_ok_path)

			for each_image_ng in images_ng:
				#print(each_image_ok)
				this_ng_image_path = os.path.join(ng, each_image_ng)
				#print(this_ng_image_path)
				pre_fix = each_image_ng.split("_")[0]
				#print(pre_fix)
				if pre_fix in test_set:
					target_ng_path = test_folder
					#print(target_path)
					shutil.copy(this_ng_image_path, target_ng_path)


