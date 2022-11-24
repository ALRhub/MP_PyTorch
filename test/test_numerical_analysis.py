import os

import torch

from mp_pytorch import util


def numerical_analysis():
    save_dir = os.path.dirname(os.path.realpath(__file__))
    file_names = util.get_file_names_in_directory(save_dir)
    file_names = [file for file in file_names if ".pt" in file]
    group_6700K = [file for file in file_names if "6700K" in file]
    group_9700K = [file for file in file_names if "9700K" in file]
    group_10900X = [file for file in file_names if "10900X" in file]
    group_3700X = [file for file in file_names if "3700X" in file]

    stats_6700K = check_error(group_9700K, group_6700K)
    print("\nIntel 9700K vs Intel 6700K", stats_6700K, sep="\n")
    stats_10900X = check_error(group_9700K, group_10900X)
    print("\nIntel 9700K vs Intel 10900X", stats_10900X, sep="\n")
    stats_3700X = check_error(group_9700K, group_3700X)
    print("\nIntel 9700K vs AMD 3700X", stats_3700X, sep="\n")


def check_error(ref_list, test_list):
    key_list = [
        "_dmp_pos", "_dmp_vel",
        "_prodmp_pc_pos_basis", "_prodmp_pc_vel_basis",
        "_prodmp_pos", "_prodmp_vel",
        "_prodmp_y_1", "_prodmp_y_2"
    ]

    error_dict = {}
    for key in key_list:
        ref_file = [file for file in ref_list if key in file][0]
        test_file = [file for file in test_list if key in file][0]
        ref_value = torch.load(ref_file)
        test_value = torch.load(test_file)
        error = torch.abs(ref_value - test_value)
        error_dict[key[1:]] = error

    stats = util.generate_many_stats(error_dict)
    return stats


if __name__ == '__main__':
    numerical_analysis()
