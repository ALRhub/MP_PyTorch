import os

from mp_pytorch import util


def numerical_analysis():
    save_dir = os.path.dirname(os.path.realpath(__file__))
    file_names = util.get_file_names_in_directory(save_dir)
    file_names = [file for file in file_names if ".pt" in file]
    group_6700K = [file for file in file_names if "6700K" in file]
    group_9700K = [file for file in file_names if "9700K" in file]
    group_3700X = [file for file in file_names if "3700X" in file]

    print(group_6700K)


if __name__ == '__main__':
    numerical_analysis()
