import os
import csv
import random
import multiprocessing
import time
import pickle
import random
from collections import Counter
import pickle

import numpy as np
from torch.utils.data import DataLoader
try:
    from torch_geometric.data import Data
except:
    pass
import torch

from utils import reshape


def encode_nodes(start, end):
    return str(start)+"_"+str(end)


_n1_attrs = None
_n2_attrs = None
_node_attrs = None
_node_types = []
_edges = []
_edge_attrs = []
_edge_types = []
_find_edge = {}

_edge_type_mat = []
_edge_attr_mat = []

_pp_edge_num = 0
_cp_edge_num = 0
_cc_edge_num = 0
_nil = None
_p_num = None
_c_num = None


# edge type
# 0: no edge
# 1: cc0 ->
# 2: cc1 <-
# 3: pp
# 4, 5, 6, 7, 8, 9, 10, 11, 12: cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9
# 13, 14, 15, 16, 17, 18, 19, 20, 21: pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9
# edge type num: 22
def _add_edge(start, end, edge_attr, edge_type):
    global _edges
    global _find_edge
    global _edge_attr_mat
    global _edge_type_mat
    global _cc_edge_num
    global _pp_edge_num
    global _cp_edge_num
    global _p_num
    global _c_num
    _edges[start].append(end)
    _edges[end].append(start)
    _find_edge[encode_nodes(start, end)] = len(_edge_attrs)
    _find_edge[encode_nodes(end, start)] = len(_edge_attrs)
    _edge_attrs.append(edge_attr)
    if edge_type == "cc":
        _edge_type_mat[start, end] = 1
        _edge_type_mat[end, start] = 2
        _edge_attr_mat[start, end] = edge_attr
        _edge_attr_mat[end, start] = edge_attr
        _cc_edge_num += 1
    elif edge_type == "pp":
        _edge_type_mat[start, end] = 3
        _edge_type_mat[end, start] = 3
        _pp_edge_num += 1
    elif edge_type == "cp":
        t = np.argmax(edge_attr)
        _edge_type_mat[start, end] = t+4
        _cp_edge_num += 1
    elif edge_type == "pc":
        t = np.argmax(edge_attr)
        _edge_type_mat[start, end] = t+13
    else:
        raise NotImplementedError()


# construct graph from raw data
def construct_graph(root):
    global _node_attrs
    global _edges
    global _edge_attrs
    global _find_edge
    global _pool
    global _edge_type_mat
    global _edge_attr_mat
    global _nil
    global _node_types
    global _n1_attrs
    global _n2_attrs
    global _p_num
    global _c_num
    _node_attrs = []
    _edges = []
    _edge_attrs = []
    _find_edge = {}
    cc_edges = parse_c_c(root)
    cp_edges = parse_c_p(root)
    c_names, label1, label2 = parse_c_label(root)
    c_values, c_map, c_map_back = parse_c_info(root)
    p_values, p_map, p_map_back = parse_p_info(root)
    c_num = len(c_map)
    _node_attrs = c_values + p_values
    _n1_attrs = np.array(c_values)
    _n2_attrs = np.array(p_values)
    _n1_attrs = np.concatenate([_n1_attrs, np.zeros(
        (_n2_attrs.shape[0], _n1_attrs.shape[1]))], axis=0)
    _n2_attrs = np.concatenate(
        [np.zeros((len(c_values), _n2_attrs.shape[1])), _n2_attrs], axis=0)
    for i in range(len(c_values)):
        _node_types.append(1)
    for i in range(len(p_values)):
        _node_types.append(2)
    node_num = len(_node_attrs)
    _edge_type_mat = np.zeros((node_num, node_num), dtype=np.int32)
    _edge_attr_mat = np.zeros((node_num, node_num))
    _edges = [[] for i in range(len(_node_attrs))]

    cp_dict = {}
    for edge in cc_edges:
        _add_edge(c_map[edge[0]], c_map[edge[1]], edge[2], "cc")
    for edge in cp_edges:
        _add_edge(c_map[edge[0]], p_map[edge[1]]+c_num, edge[2], "cp")
        _add_edge(p_map[edge[1]]+c_num, c_map[edge[0]], edge[2], "pc")
        if c_map[edge[0]] not in cp_dict:
            cp_dict[c_map[edge[0]]] = []
        cp_dict[c_map[edge[0]]].append(p_map[edge[1]]+c_num)
    for c, v in cp_dict.items():
        for x in v:
            for y in v:
                if x >= y:
                    continue
                _add_edge(x, y, None, "pp")

    _p_num = len(p_map)
    _c_num = len(c_map)

    # print("build graph summary")
    # print("cc edge num {}".format(_cc_edge_num))
    # print("pp edge num {}".format(_pp_edge_num))
    # print("cp edge num {}".format(_cp_edge_num))
    # print("p node num {}".format(_p_num))
    # print("c node num {}".format(_c_num))
    # print("v1 dim {}".format(_n1_attrs.shape))
    # print("v2 dim {}".format(_n2_attrs.shape))

# args: (node, length, n1, n2, seed, go_back_p)


def _random_walk_fn(args):
    global _edges
    global _find_edge
    global _edge_attrs
    global _node_types
    global _edge_type_mat
    global _edge_attr_mat
    global _p_num
    global _c_num
    node = args[0]
    start = node
    length = args[1]
    n1 = args[2]
    n2 = args[3]
    random.seed(args[4])
    go_back_p = args[5]
    nodes = []
    for i in range(length):
        nodes.append(node)
        if random.random() < go_back_p:
            node = start
            continue
        next_list = [v for v in _edges[node] if v != start]
        if len(next_list) == 0:
            node = start
            continue
        node = random.choice(next_list)

    # gather nodes
    nodes_ = [x for x in nodes if x != start]
    n1nodes = list(filter(lambda x: _node_types[x] == 1, nodes_))
    n2nodes = list(filter(lambda x: _node_types[x] == 2, nodes_))
    n1counts = sorted(Counter(n1nodes).items(), key=lambda x: -x[1])
    n2counts = sorted(Counter(n2nodes).items(), key=lambda x: -x[1])
    n1top_nodes = [n[0] for n in n1counts[:n1]]
    n2top_nodes = [n[0] for n in n2counts[:n2]]

    n1top_nodes.insert(0, start)
    n2top_nodes.insert(0, start)

    # build subgraph
    n1_edge_types = _edge_type_mat[n1top_nodes, :][:, n1top_nodes]
    n2_edge_types = _edge_type_mat[n2top_nodes, :][:, n2top_nodes]
    n1_attrs = _n1_attrs[n1top_nodes[1:]]
    n2_attrs = _n2_attrs[n2top_nodes[1:]]
    if n1_attrs.shape[0] < n1:
        n1_attrs = np.concatenate(
            [n1_attrs, np.zeros((n1-n1_attrs.shape[0], n1_attrs.shape[1]))], axis=0)
    if n2_attrs.shape[0] < n2:
        n2_attrs = np.concatenate(
            [n2_attrs, np.zeros((n2-n2_attrs.shape[0], n2_attrs.shape[1]))], axis=0)
    n_attrs = _node_attrs[start]
    n1_edge_attrs = _edge_attr_mat[n1top_nodes, :][:, n1top_nodes]
    n2_edge_attrs = _edge_attr_mat[n2top_nodes, :][:, n2top_nodes]

    n1_edge_types = reshape(n1_edge_types, (n1+1, n1+1))
    n2_edge_types = reshape(n2_edge_types, (n2+1, n2+1))
    n1_edge_attrs = reshape(n1_edge_attrs, (n1+1, n1+1))
    n2_edge_attrs = reshape(n2_edge_attrs, (n2+1, n2+1))
    return (n_attrs, n1_attrs,
            n2_attrs, n1_edge_types,
            n2_edge_types, n1_edge_attrs, n2_edge_attrs)


def parse_c_p(root):
    with open(os.path.join(root, "c_p.txt"), "r") as f:
        lines = f.readlines()
    edges = list(map(eval, lines))
    return edges


def parse_p_info(root):
    p_map = {}
    p_map_back = {}
    p_values = []
    with open(os.path.join(root, "p_info.txt"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            p_name = row[0]
            p_value = list(map(float, row[1:]))
            p_values.append(np.array(p_value))
            p_map[p_name] = i
            p_map_back[i] = p_name
    return p_values, p_map, p_map_back


def parse_c_info(root):
    c_map = {}
    c_map_back = {}
    c_values = []
    with open(os.path.join(root, "c_info.txt"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            c_name = row[0]
            c_value = list(map(float, row[1:]))
            c_values.append(np.array(c_value))
            c_map[c_name] = i
            c_map_back[i] = c_name
    return c_values, c_map, c_map_back


def parse_c_c(root):
    with open(os.path.join(root, "c_c.txt"), "r") as f:
        lines = f.readlines()
    edges = list(map(eval, lines))
    return edges


def parse_c_label(root):
    label1s = []
    label2s = []
    c_names = []
    with open(os.path.join(root, "c_label.txt"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            c_name = row[0]
            label1 = row[1:5]
            label2 = row[5:]
            c_names.append(c_name)
            label1s.append(label1)
            label2s.append(label2)
    label1 = np.array(label1s)
    label2 = np.array(label2s)
    return c_names, label1, label2


def process_label(root):
    cvalues, c_map, c_map_back = parse_c_info(root)
    with open(os.path.join(root, "label.txt"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            c_id = c_map[row[0]]
            print(c_id)
            exit(0)

# args: (node, length, n1, n2, seed, go_back_p)


def _random_walk_path_fn(node, length):
    global _edges
    global _find_edge
    global _edge_attrs
    global _node_types
    global _edge_type_mat
    global _edge_attr_mat
    global _p_num
    global _c_num
    start = node
    nodes = []
    for i in range(length):
        nodes.append((node, _node_types[node]))
        next_list = [v for v in _edges[node]]
        if len(next_list) == 0:
            break
        node = random.choice(next_list)
    return nodes


def data_split():
    import random
    import pickle
    c_values, c_map, c_map_back = parse_c_info("./data")
    idx = list(range(len(c_values)))
    random.shuffle(idx)
    ratio = 70
    train_size = int((ratio/100)*len(c_values))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    with open(os.path.join("./data/splits", "split_{}.pkl".format(ratio)), "wb") as f:
        pickle.dump({"train_idx": train_idx, "test_idx": test_idx}, f)


def get_data_split(train_ratio):
    with open(os.path.join("./data/splits", "split_{}.pkl".format(train_ratio)), "rb") as f:
        split = pickle.load(f)
    return split["train_idx"], split["test_idx"]


class RWCVDataset:
    def __init__(self,
                 root,
                 n1_num,
                 n2_num,
                 length,
                 cnode_ids,
                 go_back_p,
                 is_test,
                 seed=0):
        # root: path of dataset folder
        # n1_num: number of company node in neighbourhood
        # n2_num: number of people node in neighbourhood
        # length: random walk length
        # cnode_ids: id of labeled company nodes
        # go_back_p: the probability of go back to start node in random walk
        # is_test: whether testing
        self.root = root
        self.n1_num = n1_num
        self.n2_num = n2_num
        self.length = length
        self.is_test = is_test
        self.go_back_p = go_back_p
        self.cnode_ids = cnode_ids
        construct_graph(root)
        cnames, c_label1, c_label2 = parse_c_label(root)
        self.c_label1 = c_label1
        self.c_label2 = c_label2
        self.label1 = np.argmax(c_label1, axis=1)
        self.label2 = np.argmax(c_label2, axis=1)
        self.cache = {}
        self.seed = seed

    def __len__(self):
        return len(self.cnode_ids)

    # args: (node, length, n1, n2, seed, go_back_p)
    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        idx = self.cnode_ids[index]
        (n_attrs, n1_attrs,
         n2_attrs, n1_edge_types,
         n2_edge_types, n1_edge_attrs, n2_edge_attrs) = \
            _random_walk_fn((idx,
                             self.length,
                             self.n1_num,
                             self.n2_num,
                             random.randint(0, 10000000000),
                             self.go_back_p))

        # while training: we sample different neighbourhood,
        # while testing fixed neighbourhood is used.
        if self.is_test:
            self.cache[index] = ((n_attrs, n1_attrs, n2_attrs, n1_edge_types,
                                  n2_edge_types, n1_edge_attrs, n2_edge_attrs),
                                 (self.label1[idx], self.label2[idx]))
        return (n_attrs, n1_attrs, n2_attrs, n1_edge_types,
                n2_edge_types, n1_edge_attrs, n2_edge_attrs), \
            (self.label1[idx], self.label2[idx])
