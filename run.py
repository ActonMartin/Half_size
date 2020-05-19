import argparse
from agent import Agent
import time
import os
import shutil
'''

'''

class RunThis():
    def __init__(self, DefaultParam):
        self.DefaultParam = DefaultParam

    def parse_arguments(self):
        """
            Parse the command line arguments of the program.
        """
        parser = argparse.ArgumentParser(
            description='Train or test the CRNN model.')

        parser.add_argument(
            "--train_segment",
            action="store_true",
            help="Define if we wanna to train the segment net"
        )
        parser.add_argument(
            "--train_decision",
            action="store_true",
            help="Define if we wanna to train the decision net"
        )
        parser.add_argument(
            "--train_total",
            action="store_true",
            help="Define if we wanna to train the total net"
        )

        parser.add_argument(
            "--pb",
            action="store_true",
            help="Define if we wanna to get the pbmodel"
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Define if we wanna test the model"
        )
        parser.add_argument(
            "--anew",
            action="store_true",
            help="Define if we try to start from scratch  instead of  loading a checkpoint file from the save folder",
            # default=True
        )
        parser.add_argument(
            "-vr",
            "--valid_ratio",
            type=float,
            nargs="?",
            help="How the data will be split between training and testing",
            default=self.DefaultParam["valid_ratio"]
        )
        parser.add_argument(
            "-ckpt",
            "--checkPoint_dir",
            type=str,
            nargs="?",
            help="The path where the pretrained model can be found or where the model will be saved",
            default=self.DefaultParam["checkPoint_dir"])
        parser.add_argument(
            "-dd",
            "--data_dir",
            type=str,
            nargs="?",
            help="The path to the file containing the examples (training samples)",
            default=self.DefaultParam["data_dir"]
        )
        parser.add_argument(
            "-bs",
            "--batch_size",
            type=int,
            nargs="?",
            help="Size of a batch",
            default=self.DefaultParam["batch_size"]
        )
        parser.add_argument(
            "-en",
            "--epochs_num",
            type=int,
            nargs="?",
            help="How many iteration in training",
            default=self.DefaultParam["epochs_num"]
        )

        parser.add_argument(
            "-ls",
            "--label_subfix",
            type=str,
            nargs="?",
            help="How many iteration in training",
            default=self.DefaultParam["label_subfix"]
        )

        parser.add_argument(
            "-gp",
            "--gpu",
            type=str,
            nargs="?",
            help="which gpu to use",
            default=self.DefaultParam["gpu"]
        )

        parser.add_argument(
            "-nc",
            "--num_classes",
            type=int,
            nargs="?",
            help="numbers of classes",
            default=self.DefaultParam["num_classes"]
        )

        parser.add_argument(
            "-is",
            "--input_size",
            type=int,
            nargs="?",
            help="size of image input",
            default=self.DefaultParam["input_size"]
        )

        return parser.parse_args()


def delete_train_visualization():
    dir_train_visualization = "D:/Projects/head_ends_half_pix/visualization/train/"
    shutil.rmtree(dir_train_visualization)


class RunSecond():
    def __init__(self, DefaultParam):
        self.DefaultParam = DefaultParam

    def do(self):
        param = self.DefaultParam
        run_this = RunThis(param)
        # 从命令行更新参数
        args = run_this.parse_arguments()
        if not args.train_segment and not args.train_decision and not args.train_total and not args.test and not args.pb:
            print("If we are not training, and not testing, what is the point?")
        if args.train_segment:
            param["mode"] = "training"
            param["train_mode"] = "segment"
        if args.train_decision:
            param["mode"] = "training"
            param["train_mode"] = "decision"
        if args.train_total:
            param["mode"] = "training"
            param["train_mode"] = "total"
        if args.test:
            param["mode"] = "testing"
        if args.pb:
            param["mode"] = "savePb"
        if args.anew:
            param["b_restore"] = False
        param["data_dir"] = args.data_dir
        param["valid_ratio"] = args.valid_ratio
        param["batch_size"] = args.batch_size
        param["epochs_num"] = args.epochs_num
        param["checkPoint_dir"] = args.checkPoint_dir

        start = time.perf_counter()

        agent = Agent(param)
        agent.run()

        # end = time.perf_counter()
        # time_consume = end - start
        # dir_split = param["data_dir"].split('/')
        # txt_name = dir_split[3] + "_" + dir_split[4] + "_seg_dec_" + str(param["learn_rate"]) + ".txt"
        # txt_test_name = dir_split[3] + "_" + dir_split[4] + "_testing_" + str(param["learn_rate"]) + ".txt"
        # # log_folder = "E:/DATA/3_fold_half_pix/log/"
        # log_folder = param["log_time_dir"]
        # if not os.path.exists(log_folder):
        #     os.makedirs(log_folder)
        # txt_path = os.path.join(log_folder, txt_name)
        # txt_test_path = os.path.join(log_folder, txt_test_name)
        # count_path = "E:/DATA/3_fold_half_pix/count.txt"
        # 
        # if param["mode"] == "training":
        #     with open(txt_path, 'a+', encoding='utf-8') as f:
        #         f.write("{0}_".format(time_consume))
        #         if param["train_mode"] == "decision":
        #             f.write("\n")
        # 
        # if param["mode"] == "testing":
        #     with open(txt_test_path, 'a+', encoding='utf-8') as tgg:
        #         with open(count_path, 'r', encoding='utf-8') as g:
        #             number_test = int(g.read())
        #         tgg.write("{0}".format(time_consume) + "_")
        #         if number_test == 4:
        #             tgg.write("\n")
        #             self.delete_checkpoint()
        #             delete_train_visualization()
        #             number_test = -1
        #             f = open(count_path, "w")
        #             f.write(str(number_test))
        #             f.close()
        #         number_test += 1
        #         x = open(count_path, "w")
        #         x.write(str(number_test))
        #         x.close()

    def delete_checkpoint(self):
        checkpoints_files = os.listdir(self.DefaultParam["checkPoint_dir"])
        for file in checkpoints_files:
            checkpoints_files_path = os.path.join(self.DefaultParam["checkPoint_dir"], file)
            os.remove(checkpoints_files_path)


if __name__ == '__main__':
    # 导入默认参数
    # 默认参数(Notes : some params are disable )
    DefaultParam = {
        "mode": "testing",  # 模式  {"training","testing" }
        # 训练模式，{"segment":only train segment net,"decision": only train decision
        # net, "total": both}
        "train_mode": "decision",
        "data_dir":               "E:/DATA/3_fold_half_pix/fold_2/dilate_17/train_set/",  # train数据路径
        "path_which_folder_test": "E:/DATA/3_fold_half_pix/fold_2/dilate_17/", # testing的时候文件夹前部分
        "train_test_visualization": "/forloss/loss4/", # testing可视化保存路径
        "learn_rate": 1,  # 0.001

        "log_time_dir": "E:/DATA/3_fold_half_pix/log/",
        "epochs_num": 50,
        "batch_size": 1,
        "momentum": 0.9,  # 优化器参数(disable)
        "checkPoint_dir": "checkpoint",  # 模型保存路径
        "Log_dir": "Log",  # 日志打印路径
        "valid_ratio": 0,  # 数据集中用来验证的比例  (disable)
        "valid_frequency": 5,  # 每几个周期验证一次  (disable)
        "save_frequency": 2,  # 几个周期保存一次模型
        "max_to_keep": 1,  # 最多保存几个模型
        "b_restore": True,  # 导入参数
        "b_saveNG": True,  # 测试时是否保存错误的样本  (disable)
        "label_subfix": "_label.bmp",  # _label.bmp
        "gpu": "0",
        "num_classes": 2,
        "input_size": (640, 256)  # (h,w)
    }

    RunSecond(DefaultParam).do()















