import os
import pickle
import itertools
import pandas as pd
import networkx as nx
from tqdm.notebook import tqdm
from joblib import Parallel, delayed


def disjoint_all_graph(GraphList, MSMS, argumentation = False):
    if argumentation: 
        disjointed_graph_list = []
        sub_GraphLists = []
        for i in range(max(1,len(GraphList)-2), len(GraphList)+1):
            for combo in itertools.combinations(GraphList, i):
                sub_GraphLists.append(combo)
        for ithGraphList in sub_GraphLists:
            ithGraphList = list(ithGraphList)
            fragment_type = [_.graph['fragment_type'] for _ in ithGraphList]
            adduct = [_.graph['adduct'] for _ in ithGraphList]
            parent_SMILES = [_.graph['parent_SMILES'] for _ in ithGraphList]
            parent_mz = [_.graph['parent_mz'] for _ in ithGraphList]
            product_SMILES = [_.graph['product_SMILES'] for _ in ithGraphList]
            disjointed_graph = nx.disjoint_union_all(ithGraphList)
            for _ in disjointed_graph.edges(data=True): 
                if _[2]['bond_type'] == 'Fragmentation':
                    disjointed_graph.add_edges_from([(_[1], disjointed_graph.nodes(data=True)[_[1]]['atom_map_num'], {'bond_type':'Fragmentation'})])
            delete_index = [_ for _ in range(ithGraphList[0].number_of_nodes(),disjointed_graph.number_of_nodes()) if disjointed_graph.nodes(data=True)[_]['is_parent']]
            disjointed_graph.remove_nodes_from(delete_index)
            disjointed_graph = nx.disjoint_union(disjointed_graph,nx.Graph())
            disjointed_graph.graph['fragment_type'] = fragment_type
            disjointed_graph.graph['adduct'] = adduct
            disjointed_graph.graph['parent_SMILES'] = parent_SMILES
            disjointed_graph.graph['parent_mz'] = parent_mz
            disjointed_graph.graph['product_SMILES'] = product_SMILES
            disjointed_graph.graph['argumentation_ratio'] = len(ithGraphList)/len(GraphList)
            disjointed_graph_list.append(disjointed_graph)
        return disjointed_graph_list 
    else:
        fragment_type = [_.graph['fragment_type'] for _ in GraphList]
        adduct = [_.graph['adduct'] for _ in GraphList]
        parent_SMILES = [_.graph['parent_SMILES'] for _ in GraphList]
        parent_mz = [_.graph['parent_mz'] for _ in GraphList]
        product_SMILES = [_.graph['product_SMILES'] for _ in GraphList]
        disjointed_graph = nx.disjoint_union_all(GraphList)
        for _ in disjointed_graph.edges(data=True):
            if _[2]['bond_type'] == 'Fragmentation':
                disjointed_graph.add_edges_from([(_[1], disjointed_graph.nodes(data=True)[_[1]]['atom_map_num'], {'bond_type':'Fragmentation'})])
        delete_index = [_ for _ in range(GraphList[0].number_of_nodes(),disjointed_graph.number_of_nodes()) if disjointed_graph.nodes(data=True)[_]['is_parent']] # Find nodes need to delete and delete them
        disjointed_graph.remove_nodes_from(delete_index)
        disjointed_graph = nx.disjoint_union(disjointed_graph,nx.Graph())
        disjointed_graph.graph['fragment_type'] = fragment_type
        disjointed_graph.graph['adduct'] = adduct
        disjointed_graph.graph['parent_SMILES'] = parent_SMILES
        disjointed_graph.graph['parent_mz'] = parent_mz
        disjointed_graph.graph['product_SMILES'] = product_SMILES
        disjointed_graph.graph['argumentation_ratio'] = 1
        return disjointed_graph


def group_disjoint(fragment_list, MSMS, argumentation):
    product_mz = [_.graph['product_mz'] for _ in fragment_list]
    df=pd.DataFrame({
        "graph":fragment_list,
        "product_mz":product_mz
    })
    group = df.groupby("product_mz")
    if argumentation:
        list_2d = [disjoint_all_graph(list(_[1]['graph']),MSMS=MSMS,argumentation=False) for _ in group]
        flat_list = [item for sublist in list_2d for item in (sublist if isinstance(sublist, list) else [sublist])]
        return flat_list
    else: 
        return [disjoint_all_graph(list(_[1]['graph']),MSMS=MSMS,argumentation=False) for _ in group]
    

def batch_graph_grouping(step1FilePath, step2FilePath, n_jobs = -1):
    if not os.path.exists(step2FilePath): os.makedirs(step2FilePath)

    def my_task(i):
            if i not in os.listdir(step2FilePath):
                load_file = open(step1FilePath + '/' + i,"rb")
                try: fragment_list = pickle.load(load_file)
                except: return 'error'
                grouped_graph = group_disjoint(fragment_list, MSMS=None, argumentation=True)
                save_file = open(step2FilePath + '/' + i,"wb")
                pickle.dump(grouped_graph, save_file)
                save_file.close()
                return i
            else:
                return 'done'

    lst1 = os.listdir(step1FilePath)
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))