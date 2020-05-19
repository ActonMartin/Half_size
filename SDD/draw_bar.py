import os
import matplotlib.pyplot as plt
import numpy as np
'''
这个代码的作用是绘制代码运行时间的柱状图
'''
class Draw:
    def __init__(self, fold: str, learn_rate: str):
        self.path = "E:/CLONE/log_half_pix/"
        self.fold = fold
        self.learn_rate = learn_rate

    def get_filename(self):
        files = os.listdir(self.path)
        files_pre_name_list = []
        for file in files:
            file_path = os.path.join(self.path, file)
            (file_parent_path, file_name_with_extension) = os.path.split(file_path)
            (file_name, extension) = os.path.splitext(file_name_with_extension)
            files_pre_name_list.append(file_name)
        #print(files_pre_name_list)
        return files_pre_name_list

    def get_need_txt_list(self):
        seg_dec_list = []
        testing_list = []
        files_pre_name_list = self.get_filename()

        #print(files_pre_name_list)

        for each_name in files_pre_name_list:
            file_name_split = each_name.split('_')
            if file_name_split[1] == self.fold and file_name_split[-1] == self.learn_rate and file_name_split[4] == 'seg':
                seg_dec_list.append(each_name)
            if file_name_split[1] == self.fold and file_name_split[-1] == self.learn_rate and file_name_split[4] == 'testing':
                testing_list.append(each_name)
        seg_dec_list.sort(key=lambda x: int(x.split('_')[3]))
        testing_list.sort(key=lambda x: int(x.split('_')[3]))
        return seg_dec_list, testing_list

    def get_seg_dec_testing_mean(self):
        all_seg_mean = []
        all_dec_mean = []
        all_testing_mean = []
        seg_dec_list, testing_list = self.get_need_txt_list()
        for item in seg_dec_list:
            this_seg_list = []
            this_dec_list = []
            txt_path = self.path + item + '.txt'
            with open(txt_path,mode='r') as kkk:
                lines = kkk.readlines()
                for line in lines:
                    line = line.strip('\n')
                    this_seg_list.append(line.split('_')[0])
                    this_dec_list.append(line.split('_')[1])
            this_seg_list = list(map(eval, this_seg_list))
            this_dec_list = list(map(eval, this_dec_list))
            average_seg = sum(this_seg_list) / len(this_seg_list)
            average_dec = sum(this_dec_list) / len(this_dec_list)
            all_seg_mean.append(average_seg)
            all_dec_mean.append(average_dec)
        for item in testing_list:
            this_testing_list = []
            txt_path = self.path + item + '.txt'
            with open(txt_path,mode='r') as kkk:
                lines = kkk.readlines()
                for line in lines:
                    line = line.strip('\n')
                    line_list = line.split('_')
                    for i in range(5):
                        this_testing_list.append(line_list[i])
            this_testing_list = list(map(eval, this_testing_list))
            average_testing = sum(this_testing_list) / len(this_testing_list)
            all_testing_mean.append(average_testing)
        print('-'*20)
        print(all_seg_mean)
        print(all_dec_mean)
        print(all_testing_mean)
        print('-' * 20)
        return all_seg_mean, all_dec_mean, all_testing_mean

    def drawpillar(self):
        n_groups = 5
        all_seg_mean, all_dec_mean, all_testing_mean = self.get_seg_dec_testing_mean()
        # plt.figure(figsize=(6.4, 4.8), dpi=300)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
        index = np.arange(n_groups)
        bar_width = 0.15
        opacity = 0.4
        rects1 = plt.bar(index, all_seg_mean, bar_width, alpha=opacity, color='b', label='segmentation')
        rects2 = plt.bar(index + bar_width, all_dec_mean, bar_width, alpha=opacity, color='g', label='decision')
        rects2 = plt.bar(index + 2*bar_width, all_testing_mean, bar_width, alpha=opacity, color='r', label='test')

        plt.xlabel('dilate', fontsize=20)
        plt.ylabel('time/s', fontsize=20)
        plt.title('fold_' + self.fold + '_learn_rate_' + self.learn_rate, fontsize=20)
        plt.xticks(index + bar_width, ('0', '5', '9', '13', '17'), fontsize=16)
        plt.yticks(np.arange(0, 253, 50), fontsize=16)
        plt.ylim(0, 253)
        plt.legend(fontsize=12, framealpha=0.8, frameon=False)

        plt.tight_layout()
        img_name = 'fold_' + self.fold + '_learn_rate_' + self.learn_rate + '.svg'
        img = plt.savefig(img_name)
        # plt.show()
        return img


if __name__ == '__main__':
    fold_list = ['0', '1', '2']
    learn_rate_list = ['0.01', '0.1', '0.5','1']
    for fold in fold_list:
        for rate in learn_rate_list:
            draw = Draw(fold, rate)
            #get_mean = draw.get_seg_dec_testing_mean()
            show_pic = draw.drawpillar()



