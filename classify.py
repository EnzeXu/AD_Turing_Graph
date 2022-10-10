import numpy as np
import os
import matplotlib.pyplot as plt

from classify_methods import *
from utils import generate_node_file


def generate_rank(x):
    x_expand = [[x[i], i] for i in range(len(x))]
    x_expand.sort(key=lambda xx: xx[0])
    # print(x_expand)
    rank = [-1] * len(x)
    for i in range(len(x)):
        rank[x_expand[i][1]] = i
    return rank


def first_large_index(x, target):
    rank = generate_rank(x)
    best = 99999
    best_index = -1
    for i in range(len(x)):
        if target <= x[i] < best:
            best = x[i]
            best_index = i
    return rank[best_index], rank[best_index], len(x) - rank[best_index]


def match(A, T, threshold_A, threshold_T, mid_rate=0.2):
    assert len(A) == len(T)
    n = len(A)
    # print("mean of A: {}".format(np.mean(A)))
    # print("mean of T: {}".format(np.mean(T)))
    # x = range(n)

    # threshold_A = np.mean(A)
    # threshold_T = np.mean(T)

    # mid_rate = 0.2

    A_rank = generate_rank(A)
    T_rank = generate_rank(T)

    A_class, T_class = np.zeros(n), np.zeros(n)
    result_mat = np.zeros([3, 3])

    A_mid_index, A_left, A_right = first_large_index(A, threshold_A)
    # print(A_mid_index, A_left, A_right)
    T_mid_index, T_left, T_right = first_large_index(T, threshold_T)
    # print(T_mid_index, T_left, T_right)
    for i in range(n):
        if A_rank[i] < A_mid_index - A_left * mid_rate:
            A_class[i] = 0
        elif A_rank[i] >= A_mid_index + A_right * mid_rate:
            A_class[i] = 2
        else:
            A_class[i] = 1

        if T_rank[i] < T_mid_index - T_left * mid_rate:
            T_class[i] = 0
        elif T_rank[i] >= T_mid_index + T_right * mid_rate:
            T_class[i] = 2
        else:
            T_class[i] = 1

        result_mat[int(A_class[i])][int(T_class[i])] += 1

    # print(list(A_class).count(0), list(A_class).count(1), list(A_class).count(2))
    # print(list(T_class).count(0), list(T_class).count(1), list(T_class).count(2))
    # print(result_mat)

    print("\tT-\tT_m\tT+")
    Ts = ["A-", "A_m", "A+"]
    for i in range(3):
        print(Ts[i], end="")
        for j in range(3):
            print("\t{}".format(int(result_mat[i][j])), end="")
        print()
    match_rate = (result_mat[2][0] + result_mat[1][1] + result_mat[0][2]) / n
    print("match rate: {0:d} / {1:d} = {2:.4f}%".format(int(result_mat[2][0] + result_mat[1][1] + result_mat[0][2]), n,
                                                        match_rate * 100))
    return A_class, T_class


def classify(method):
    threshold = np.load("data_brain/row/overall_threshold.npy")
    threshold_A = threshold[0]
    threshold_T = threshold[1]

    for one_type in ["CN", "MCI", "AD"]:
        A = np.load("data/average_node/A_{}.npy".format(one_type))
        T = np.load("data/average_node/T_{}.npy".format(one_type))
        A_class, T_class = match(A, T, threshold_A, threshold_T, method["mid_rate"])
        n = len(A)
        classification = np.zeros(n)
        for i in range(n):
            assert A_class[i] in method["class_input_list"] and T_class[i] in method["class_input_list"]
            classification[i] = method["class_rule"][int(A_class[i])][int(T_class[i])]
            assert classification[i] in method["class_output_list"]
        np.save("data_brain/BrainNet/class/node_class_{}_{}.npy".format(method["name"], one_type), classification)
        print(one_type, classification)


def one_time_generate_classified_nodes():
    class_path = "data_brain/BrainNet/class/"
    files = os.listdir(class_path)
    files = [item for item in files if ".npy" in item]
    for one_file in files:
        color_array = np.load(os.path.join(class_path, one_file))
        size_array = np.zeros(160)
        generate_node_file(color_array, size_array, "data_brain/BrainNet/node/{}.node".format(one_file.replace(".npy", "")))


if __name__ == "__main__":
    classify(method1)
    classify(method2)
    # one_time_generate_classified_nodes()
    pass
