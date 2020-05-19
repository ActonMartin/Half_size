import os
import matplotlib.pyplot as plt
import numpy as np
'''
这个代码的作用是绘制分类折线图
'''


class DrawLineChart():
	def __init__(self, fold:str, learn_rate:str, ng, ok, ng_list:list, ok_list:list):
		self.fold = fold
		self.learn_rate = learn_rate
		self.ng = ng # 真实的NG数量
		self.ok = ok # 真实的OK数量
		self.ng_list = ng_list
		self.ok_list = ok_list
		self.dilate = [0, 5, 9, 13, 17]

	def drawpillar(self):
		# plt.figure(figsize=(6.4, 4.8), dpi=300)
		fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
		y_ticks = np.arange(0, 140, 10)

		rects1 = plt.plot(self.dilate,self.ng_list,color="blue",linewidth=2,linestyle=':',label='NG', marker='*')
		rects1 = plt.plot(self.dilate,self.ok_list,color="green",linewidth=2,linestyle=':',label='OK', marker='*')
		rects1 = plt.plot(self.dilate,self.ng,color="black",linewidth=2,linestyle='-',label='real NG', marker='.')
		rects1 = plt.plot(self.dilate,self.ok,color="black",linewidth=2,linestyle='-',label='real OK', marker='.')

		plt.xlabel('dilate', fontsize=20)
		plt.ylabel('images', fontsize=20)
		plt.title('fold_' + self.fold + '_learn_rate_' + self.learn_rate, fontsize=20)
		plt.xticks(self.dilate , ('0', '5', '9', '13', '17'), fontsize=16)

		plt.yticks(y_ticks,fontsize=16)
		plt.ylim(0, 140)
		plt.legend(fontsize=12, framealpha=0.8, frameon=True)

		plt.tight_layout()
		img_name = 'fold_' + self.fold + '_learn_rate_' + self.learn_rate + '.svg'
		img = plt.savefig(img_name)
		# plt.show()
		return img


if __name__ == "__main__":
	ng_list_fold_0 = [17.8, 17.6, 18.6, 18.4, 18.2]
	ok_list_fold_0 = [117.2, 117.4, 117, 116.6, 116.8]
	ng_list_fold_1 = [17, 17, 17, 17, 16.8]
	ok_list_fold_1 = [119, 119, 119, 119, 119.2]
	ng_list_fold_2 = [13.8, 13.6, 13.6, 14.2, 13.8]
	ok_list_fold_2 = [114.2, 114.4, 114.4, 113.8, 114.2]
	DrawLineChart('0', '0.01', [18]*5, [117]*5, ng_list_fold_0, ok_list_fold_0).drawpillar()
	DrawLineChart('1', '0.01', [18]*5, [118]*5, ng_list_fold_1, ok_list_fold_1).drawpillar()
	DrawLineChart('2', '0.01', [16]*5, [112]*5, ng_list_fold_2, ok_list_fold_2).drawpillar()

	ng_list_fold_0_2 = [18.8, 18.6, 19.2, 18.6, 18.8]
	ok_list_fold_0_2 = [116.2, 116.4, 115.8, 116.4, 116.2]
	ng_list_fold_1_2 = [17, 17, 17, 17, 17.2]
	ok_list_fold_1_2 = [119, 119, 119, 119, 118.8]
	ng_list_fold_2_2 = [14.4, 14.4, 14.2, 14, 14]
	ok_list_fold_2_2 = [113.6, 113.6, 113.8, 114, 114]
	DrawLineChart('0', '0.1', [18] * 5, [117] * 5, ng_list_fold_0_2, ok_list_fold_0_2).drawpillar()
	DrawLineChart('1', '0.1', [18] * 5, [118] * 5, ng_list_fold_1_2, ok_list_fold_1_2).drawpillar()
	DrawLineChart('2', '0.1', [16] * 5, [112] * 5, ng_list_fold_2_2, ok_list_fold_2_2).drawpillar()

	ng_list_fold_0_3 = [74.8, 18.4, 95.6, 33.6, 33.4]
	ok_list_fold_0_3 = [60.2, 116.6, 39.4, 101.4, 101.6]
	ng_list_fold_1_3 = [40.2, 34.4, 36.4, 40.8, 13]
	ok_list_fold_1_3 = [95.8, 101.6, 99.6, 95.2, 123]
	ng_list_fold_2_3 = [14, 24.8, 26.6, 26.8, 18.2]
	ok_list_fold_2_3 = [114, 103.2, 101.4, 101.2, 109.8]
	DrawLineChart('0', '0.5', [18] * 5, [117] * 5, ng_list_fold_0_3, ok_list_fold_0_3).drawpillar()
	DrawLineChart('1', '0.5', [18] * 5, [118] * 5, ng_list_fold_1_3, ok_list_fold_1_3).drawpillar()
	DrawLineChart('2', '0.5', [16] * 5, [112] * 5, ng_list_fold_2_3, ok_list_fold_2_3).drawpillar()

	ng_list_fold_0_4 = [134.8, 84.4, 57.2, 107.8, 90]
	ok_list_fold_0_4 = [0.2, 50.6, 77.8, 27.2, 45]
	ng_list_fold_1_4 = [136, 136, 125.2, 109, 83.2]
	ok_list_fold_1_4 = [0, 0, 10.8, 27, 52.8]
	ng_list_fold_2_4 = [92.6, 128, 79.6, 84.8, 78]
	ok_list_fold_2_4 = [35.4, 0, 48.4, 46.2, 50]
	DrawLineChart('0', '1', [18] * 5, [117] * 5, ng_list_fold_0_4, ok_list_fold_0_4).drawpillar()
	DrawLineChart('1', '1', [18] * 5, [118] * 5, ng_list_fold_1_4, ok_list_fold_1_4).drawpillar()
	DrawLineChart('2', '1', [16] * 5, [112] * 5, ng_list_fold_2_4, ok_list_fold_2_4).drawpillar()
