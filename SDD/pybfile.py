import pickle

PATH_TO_FILE = r'E:\DATA\KolektorSDD-training-splits\split_small_20.pyb'
with open(PATH_TO_FILE, 'rb') as f:
	[train_split, test_split, all] = pickle.load(f)
print(train_split)
print(test_split)
print(all)
