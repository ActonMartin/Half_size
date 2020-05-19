import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
'''
这个代码的作用是绘制loss
'''

class Draw:
    def __init__(self, fold: str, dilate: str):
        self.path = "E:/CLONE/log_half_loss/"
        self.fold = fold
        self.dilate = dilate

    def get_filename(self):
        files = os.listdir(self.path)
        files_pre_name_list = []
        for file in files:
            file_path = os.path.join(self.path, file)
            (file_parent_path, file_name_with_extension) = os.path.split(file_path)
            (file_name, extension) = os.path.splitext(file_name_with_extension)
            files_pre_name_list.append(file_name)
        # print(files_pre_name_list)
        return files_pre_name_list

    def get_need_txt_list(self):
        seg_list = []
        dec_list = []
        files_pre_name_list = self.get_filename()

        #print(files_pre_name_list)

        for each_name in files_pre_name_list:
            file_name_split = each_name.split('_')
            if file_name_split[1] == self.fold and file_name_split[3] == self.dilate and file_name_split[4] == 'seg':
                seg_list.append(each_name)
            if file_name_split[1] == self.fold and file_name_split[3] == self.dilate and file_name_split[4] == 'dec':
                dec_list.append(each_name)
        seg_list.sort(key=lambda x: float(x.split('_')[-1]))
        dec_list.sort(key=lambda x: float(x.split('_')[-1]))
        print("seg_list", seg_list)
        print("dec_list", dec_list)
        return seg_list, dec_list

    def get_loss(self):
        seg_list, dec_list = self.get_need_txt_list()

        def get_loss_list(a_list, learn_rate):
            a_loss_list = []
            for item in a_list:
                if item.split('_')[-1] == learn_rate:
                    txt_path = self.path + item + '.txt'
                    with open(txt_path, 'r') as kkk:
                        lines = kkk.readlines()[1:]  # 去除第一行，从第二行开始读取
                        for line in lines:
                            this_loss = float(line.split(':')[-1].strip())
                            a_loss_list.append(this_loss)
            return a_loss_list

        seg_loss_1 = get_loss_list(seg_list, '0.01')
        dec_loss_1 = get_loss_list(dec_list, '0.01')
        seg_loss_2 = get_loss_list(seg_list, '0.1')
        dec_loss_2 = get_loss_list(dec_list, '0.1')
        seg_loss_3 = get_loss_list(seg_list, '0.5')
        dec_loss_3 = get_loss_list(dec_list, '0.5')
        seg_loss_4 = get_loss_list(seg_list, '1')
        dec_loss_4 = get_loss_list(dec_list, '1')
        print(len(seg_loss_4))
        return seg_loss_1, dec_loss_1, seg_loss_2, dec_loss_2,seg_loss_3, dec_loss_3, seg_loss_4, dec_loss_4

    def draw_subplot(self):
        seg_loss_1, dec_loss_1, seg_loss_2, dec_loss_2,seg_loss_3, dec_loss_3, seg_loss_4, dec_loss_4 = self.get_loss()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6.4, 4.8), dpi=300)
        epoch = np.arange(1, 51, 1)

        ax1.plot(epoch, seg_loss_1, color="blue", linewidth=1, linestyle=':', label='segmentation', marker='.',
                 markersize=3)
        ax1.plot(epoch, dec_loss_1, color="black", linewidth=1, linestyle=':', label='decision', marker='.',
                 markersize=3)
        ax1.set_xlabel('epoch', fontsize=8)
        ax1.set_ylabel('loss', fontsize=8)
        ax1.set_title('fold_' + self.fold + '_learn_rate_0.01')
        ax1.set_yticks(np.arange(0, 60, 5))
        ax1.xaxis.set_major_locator(MultipleLocator(5))
        ax1.set_ylim((0, 60))
        ax1.legend(fontsize=12, framealpha=0.8, frameon=True)

        ax2.plot(epoch, seg_loss_2, color="blue", linewidth=1, linestyle=':', label='segmentation', marker='.',
                 markersize=3)
        ax2.plot(epoch, dec_loss_2, color="black", linewidth=1, linestyle=':', label='decision', marker='.',
                 markersize=3)
        ax2.set_xlabel('epoch', fontsize=8)
        ax2.set_ylabel('loss', fontsize=8)
        ax2.set_title('fold_' + self.fold + '_learn_rate_0.1')
        ax2.set_yticks(np.arange(0, 60, 5))
        ax2.xaxis.set_major_locator(MultipleLocator(5))
        ax2.set_ylim((0, 60))
        ax2.legend(fontsize=12, framealpha=0.8, frameon=True)

        ax3.plot(epoch, seg_loss_3, color="blue", linewidth=1, linestyle=':', label='segmentation', marker='.',
                 markersize=3)
        ax3.plot(epoch, dec_loss_3, color="black", linewidth=1, linestyle=':', label='decision', marker='.',
                 markersize=3)
        ax3.set_xlabel('epoch', fontsize=8)
        ax3.set_ylabel('loss', fontsize=8)
        ax3.set_title('fold_' + self.fold + '_learn_rate_0.5')
        ax3.set_yticks(np.arange(0, 60, 5))
        ax3.xaxis.set_major_locator(MultipleLocator(5))
        ax3.set_ylim((0, 60))
        ax3.legend(fontsize=12, framealpha=0.8, frameon=True)

        ax4.plot(epoch, seg_loss_4, color="blue", linewidth=1, linestyle=':', label='segmentation', marker='.',
                 markersize=3)
        ax4.plot(epoch, dec_loss_4, color="black", linewidth=1, linestyle=':', label='decision', marker='.',
                 markersize=3)
        ax4.set_xlabel('epoch', fontsize=8)
        ax4.set_ylabel('loss', fontsize=8)
        ax4.set_title('fold_' + self.fold + '_learn_rate_1')
        ax4.set_yticks(np.arange(0, 60, 5))
        ax4.xaxis.set_major_locator(MultipleLocator(5))
        ax4.set_ylim((0, 60))
        ax4.legend(fontsize=12, framealpha=0.8, frameon=True)
        plt.tight_layout()
        img_name = 'fold_' + self.fold + '_dilate_' + self.dilate + '.svg'
        img = plt.savefig(img_name)
        # plt.show()
        plt.close()
        return img

    def draw_out(self):
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
        epoch = np.arange(0, 50, 1)
        seg_loss, dec_loss = self.get_loss()
        rects1 = plt.plot(epoch, seg_loss, color="blue", linewidth=2, linestyle=':', label='segmentation', marker='*',
                          markersize=1)
        rects2 = plt.plot(epoch, dec_loss, color="black", linewidth=2, linestyle=':', label='decision', marker='*',
                          markersize=1)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.title('fold_' + self.fold + '_learn_rate_' + self.learn_rate)
        plt.yticks(np.arange(0, 55, 5), fontsize=16)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        plt.ylim((0, 55))
        plt.legend(fontsize=12, framealpha=0.8, frameon=True)
        img_name = 'fold_' + self.fold + '_learn_rate_' + self.learn_rate + '.svg'
        img = plt.savefig(img_name)
        # plt.show()
        return img


if __name__ == "__main__":
    fold = ['0', '1', '2']
    dilate = ['0', '5', '9', '13', '17']
    for i in fold:
        for j in dilate:
            Draw(fold=i, dilate=j).draw_subplot()

