# import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import os

# from pyvis.network import Network

from data.column_names import *

# def random_edge(n_max):
#     s = random.randint(1, n_max)
#     d = random.randint(1, n_max)
#     return [s, d]
#
# def random_undirected_network(n_node, n_edge=None):
#     if not n_edge:
#         n_edge = int(n_node * (n_node - 1) / 6)
#     node_list = [i + 1 for i in range(n_node)]
#     lc = LinearColor()
#     value_list = [1 for i in range(n_node)]
#     color_list = [lc.trans(random.random()) for i in range(n_node)]
#     z = n_edge
#     edge_list = []
#     while z > 0:
#         one_edge = random_edge(n_node)
#         if one_edge[0] != one_edge[1] and one_edge not in edge_list and [one_edge[1], one_edge[0]] not in edge_list:
#             edge_list.append(one_edge)
#             z -= 1
#     g = Network(height="1500px", width="100%")
#     g.add_nodes(node_list, color=color_list, value=value_list)
#     g.add_edges(edge_list)
#     g.set_edge_smooth("dynamic")
#     # g.toggle_physics(True)
#     # "physics": {
#     #     "enabled": false,
#     options = """
# {
#   "edges": {
#     "color": {
#       "opacity": 0.15
#     },
#     "smooth": false
#   },
#   "physics": {
#     "enabled": false,
#     "barnesHut": {
#       "avoidOverlap": 0.91
#     }
#   }
# }
#     """
#
#     # g.set_options(options)
#     g.barnes_hut()
#
#     return g
#
# class LinearColor:
#     def __init__(self, left="#ffdcdc", right="#7f0000"):
#         self.left_pair = self.decode(left)
#         self.right_pair = self.decode(right)
#         self.diff_pair = [self.right_pair[i] - self.left_pair[i] for i in range(3)]
#
#     @staticmethod
#     def decode(color_str):
#         return [int("0x" + color_str[2 * i + 1: 2 * i + 3], 16) for i in range(3)]
#
#     @staticmethod
#     def encode(color_pair):
#         return "#" + "".join([str(hex(item))[2:].zfill(2) for item in color_pair])
#
#     def trans(self, val):
#         assert 0 <= val <= 1.0, "Value should be in [0, 1]"
#         new_pair = [int(self.left_pair[i] + self.diff_pair[i] * val) for i in range(3)]
#         return self.encode(new_pair)
#
#
# def test():
#     g = Network()
#     g.add_nodes(["a", "b", "c"],
#                 # value=[10, 100, 400],
#                 # title=["a", "b", "c"],
#                 # x=[21.4, 54.2, 11.2],
#                 # y=[100.2, 23.54, 32.1],
#                 # label=['NODE 1', 'NODE 2', 'NODE 3'],
#                 # color=['#00ff1e', '#162347', '#dd4b39']
#                 color=[(255, 0, 0), (128,128,128), (0,0,255)],
#                 smooth=[False, False, False]
#                 )
#     g.add_edges([["a", "b"], ["a", "c"]])
#     g.set_edge_smooth("straightCross")
#     g.show_buttons(filter_=["nodes", "edges", "physics"])
#     g.show("test.html")
#     # df = pd.DataFrame()
#     # df["src"] = ["a", "a", "b", "b", "c"]
#     # df["dst"] = ["b", "c", "c", "d", "d"]
#     # df["weight"] = [1.0, 1.0, 1.0, 1.0, 1.0]
#     # g = nx.from_pandas_edgelist(df, "src", "dst", "weight")
#     # # nx.draw(g)
#     # print(g)
#     # nx.set_node_attributes(g, {"a": 1.5, "b": 2.0, "c": 1.5, "d": 3.0}, name="val")
#     # net = Network()
#     # net.from_nx(g)
#     # net.show("example.html")


def one_time_load_data(data_a_path="data/271amyloid.csv", data_t_path="data/271tau.csv"):
    data_a = pd.read_csv(data_a_path)
    data_t = pd.read_csv(data_t_path)
    data_a = data_a[COLUMN_NAMES + TITLE_NAMES]
    data_t = data_t[COLUMN_NAMES + TITLE_NAMES]

    df = data_a
    save_path = "data/average_node/A_{}.npy"
    collection = np.zeros((3, 160))
    counts = np.zeros(3)
    for index, row in df.iterrows():
        label = None
        for one_key in LABELS:
            if row["DX"] in LABELS[one_key]:
                label = one_key
                counts[LABEL_ID[label]] += 1
                break
        if not label:
            continue
        for i in range(160):
            collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES[i]])
    for one_key in LABELS:
        avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
        print(avg)
        np.save(save_path.format(one_key), avg)
    print(counts)
    a_overall_threshold = np.sum(collection) / np.sum(counts) / 160

    df = data_t
    save_path = "data/average_node/T_{}.npy"
    collection = np.zeros((3, 160))
    counts = np.zeros(3)
    for index, row in df.iterrows():
        label = None
        for one_key in LABELS:
            if row["DX"] in LABELS[one_key]:
                label = one_key
                counts[LABEL_ID[label]] += 1
                break
        if not label:
            continue
        for i in range(160):
            collection[LABEL_ID[label]][i] += float(row[COLUMN_NAMES[i]])
    for one_key in LABELS:
        avg = collection[LABEL_ID[one_key], :] / counts[LABEL_ID[one_key]]
        print(avg)
        np.save(save_path.format(one_key), avg)
    print(counts)
    t_overall_threshold = np.sum(collection) / np.sum(counts) / 160

    overall_threshold = np.asarray([a_overall_threshold, t_overall_threshold])
    np.save("data/average_network/overall_threshold.npy", overall_threshold)
    print(overall_threshold)


def generate_node_file(color_array, size_array, save_path, length=160):
    assert len(size_array) == len(color_array) == length
    df_dress = pd.read_excel("data_brain/BrainNet/destriux_160.xlsx")
    with open(save_path, "w") as f:
        for i in range(length):
            f.write("{} {} {} {} {} {}\n".format(
                str(df_dress["x"][i]),
                str(df_dress["y"][i]),
                str(df_dress["z"][i]),
                str(color_array[i]),
                str(size_array[i]),
                str(df_dress["name"][i]),
            ))


def norm_row(mat):
    sum_sqrt = np.sqrt(np.sum(mat * mat, axis=1)).reshape(-1, 1)
    sum_sqrt += (sum_sqrt == 0) * 0.0001
    return mat / sum_sqrt


def norm_symmetric(mat):
    # print(mat.shape)
    mat = norm_row(mat)
    return (mat + mat.swapaxes(0, 1)) / 2.0





def load_one_network(file_name):
    return np.loadtxt(file_name)


def generate_network_avg(directory_path, save_path="data/average_network/network_avg.npy"):
    names = os.listdir(directory_path)
    names = [item for item in names if "S" in item]
    print("number of networks:", len(names))
    network_sum = np.zeros([3, 160, 160], dtype=float)
    # names = names[:5]
    # for i in range(3):
    with open("data/dx_dictionary.pkl", "rb") as f:
        dx_dictionary = pickle.load(f)
    bad_count = 0
    count = np.zeros([3])
    for item in names:
        if item[:10] not in dx_dictionary:
            bad_count += 1
            print("PTID {} does not match! (count: {})".format(item[:10], bad_count))
        else:
            network_sum[LABEL_ID[dx_dictionary[item[:10]]]] += norm_symmetric(load_one_network(os.path.join(directory_path, item)))
            count[LABEL_ID[dx_dictionary[item[:10]]]] += 1
    for item in ["CN", "MCI", "AD"]:
        network_avg = network_sum[LABEL_ID[item]] / count[LABEL_ID[item]]
        print(network_avg.shape)
        print("item: {} count: {}".format(item, count[LABEL_ID[item]]))
        # print(network_avg)
        assert (network_avg == network_avg.transpose()).all()
        np.save(save_path.replace(".npy", "_{}.npy".format(item)), network_avg)
    # return network_avg


def generate_network_laplacian(from_path, save_path="data/average_network/network_laplacian.npy", threshold=0.2):
    # print(network)
    network = np.load(from_path)
    assert network.shape[0] == network.shape[1]
    # network = network / np.max(network)
    network = network - (network <= threshold) * network
    print("zero:", np.count_nonzero(network == 0))
    print("non-zero:", network.shape[0] * network.shape[1] - np.count_nonzero(network == 0))
    print("< {}:".format(threshold), np.count_nonzero(network <= threshold))
    print("> {}:".format(threshold), np.count_nonzero(network > threshold))

    degree = np.diag(np.sum(network, axis=0))
    laplacian = degree - network
    # degree_out = np.diag(np.sum(network, axis=1))
    # laplacian_out = degree_out - network
    # print("laplacian_in:")
    # print(laplacian_in)
    # print("laplacian_out:")
    # print(laplacian_out)
    np.save(save_path, laplacian)
    # np.save(save_path_out, laplacian_out)
    return laplacian


def one_time_generate_six_average_nodes():
    for one_name in ["A_CN", "A_MCI", "A_AD", "T_CN", "T_MCI", "T_AD"]:
        color_array = np.load("data/average_node/{}.npy".format(one_name))
        color_array = color_array / np.max(color_array)
        # print(size_array)
        # color_array = [1 if color_array[i] >= np.median(color_array) else 0 for i in range(len(color_array))]
        print(color_array)
        size_array = np.zeros(160)  # np.load("data_brain/row/A_AD.npy")
        generate_node_file(color_array, size_array, "data_brain/BrainNet/node/{}.node".format(one_name))


def one_time_build_patient_dx_dictionary():
    path = "data/FDG_Full.xlsx"
    df = pd.read_excel(path)[["DX", "PTID"]]
    dx_dic = dict()
    for index, row in df.iterrows():
        if row["DX"] in LABELS["CN"]:
            dx_dic[row["PTID"]] = "CN"
        elif row["DX"] in LABELS["MCI"]:
            dx_dic[row["PTID"]] = "MCI"
        elif row["DX"] in LABELS["AD"]:
            dx_dic[row["PTID"]] = "AD"
        else:
            print("matching failed in row {}, PTID = {}, DX = {}".format(index, row["PTID"], row["DX"]))
    with open("data/dx_dictionary.pkl", "wb") as f:
        pickle.dump(dx_dic, f)


def one_time_calculate_threshold_classify_nodes():
    pass






if __name__ == "__main__":
    # generate_network_avg("data/network")
    # generate_network_laplacian("data/average_network/network_avg_CN.npy", "data/average_network/network_laplacian_CN.npy")
    # generate_network_laplacian("data/average_network/network_avg_MCI.npy", "data/average_network/network_laplacian_MCI.npy")
    # generate_network_laplacian("data/average_network/network_avg_AD.npy", "data/average_network/network_laplacian_AD.npy")
    one_time_load_data()
    # # net = np.asarray([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    # generate_network_laplacian(net, "data/average_network/network_laplacian.npy", 0.01)
    # one_time_build_patient_dx_dictionary()

    # generate_network_avg("data/network")
    # load_data()
    # mat = np.asarray([[0.6, 0.8], [1, 0]])
    # assert (mat == mat.transpose()).all()
    # print(norm_symmetric(mat))

    # print(np.zeros(160))
    # graph = random_undirected_network(50, 150)
    # graph.show("test.html")
    # graph.show_buttons(filter_=["nodes", "edges", "physics"])

    # for i in range(20):
    #     print(random.random())
    # print(str(hex(23)))
    # print(int("0xde"))
    # test()
    # print(hex(220))
    # print(hex(127))
    pass
