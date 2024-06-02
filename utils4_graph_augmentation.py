import os
import copy
import pickle
import numpy as np
import networkx as nx
from itertools import product
from tqdm.notebook import tqdm
from joblib import Parallel, delayed


# networkx
def graph_augmentation(raw_Graph, repeat = 2, augmentation_ratio = 0.6, with_raw_Graph = True):
    copyed_Graph = copy.deepcopy(raw_Graph)
    Fragmentation_edges = [(_[0],_[1]) for _ in copyed_Graph.edges(data=True) if _[2]['bond_type'] == 'Fragmentation']
    copyed_Graph.remove_edges_from(Fragmentation_edges)
    copyed_Graph.edges(data=True)

    sub_Graphs_index = list(nx.connected_components(copyed_Graph))
    sub_Graphs_index = [list(_) for _ in sub_Graphs_index if copyed_Graph.nodes[list(_)[0]]['is_parent'] == False]
    sub_Graphs_index = list(product(sub_Graphs_index, repeat = min(repeat, int(len(sub_Graphs_index) * (1 - augmentation_ratio)))))
    sub_Graphs_index = [(set([i for k in _ for i in k])) for _ in sub_Graphs_index]
    sub_Graphs_index = [list(t) for t in set(tuple(_) for _ in sub_Graphs_index)]

    if with_raw_Graph: augmented_Graphs = [raw_Graph]
    else: augmented_Graphs = []
    if len(sub_Graphs_index[0]) > 0:
        for ithGraph in sub_Graphs_index:
            ith_copyed_Graph = copy.deepcopy(raw_Graph)
            ith_copyed_Graph.remove_nodes_from(ithGraph)
            augmented_Graphs += [nx.disjoint_union(ith_copyed_Graph,nx.Graph())]
    return augmented_Graphs


def batch_graph_augmentation(step3FilePath, step4FilePath, n_jobs = -1):
    if not os.path.exists(step4FilePath): os.makedirs(step4FilePath)

    def my_task(i):
        load_file = open(step3FilePath + '/' + i,"rb")
        try: fragment_list = pickle.load(load_file)
        except: return 'error'
        augmented_fragment_list = [graph_augmentation(_) if _.graph['intensity_regression1'] else [_] for _ in fragment_list] # Data augmentation
        augmented_fragment_list = [i for k in augmented_fragment_list for i in k]
        save_file = open(step4FilePath + '/' + i,"wb")
        pickle.dump(augmented_fragment_list, save_file)
        save_file.close()
        return i
    
    lst1 = os.listdir(step3FilePath)
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))
