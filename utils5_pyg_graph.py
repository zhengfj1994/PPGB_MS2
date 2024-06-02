import os
import copy
import torch
import pickle
import numpy as np
import networkx as nx
from rdkit import Chem
from torch import Tensor
from utils import pickle_load
from itertools import product
from tqdm.notebook import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from torch_geometric.data import Data
from torch_geometric.data import HeteroData


def one_hot_k_encode(x, permitted_list):
    """
    x: used to convert to one-hot vector
    premitted_list: predefined list
    """
    if x not in permitted_list: x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def process(G, type, is_homo = False):
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    data = defaultdict(list)

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                pass
    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    permitted_list_of_atoms = [6, 7, 8, 16, 15, 9, 17, 35, 53,'unknown']
    atomic_num_one_hot = np.zeros((len(data.atomic_num), len(permitted_list_of_atoms)))
    for _ in range((len(data.atomic_num))):
        atomic_num_one_hot[_, :] = one_hot_k_encode(data.atomic_num[_], permitted_list_of_atoms)
    atomic_num_one_hot = torch.tensor(atomic_num_one_hot).to(torch.float32)
    
    formal_charge_one_hot = np.zeros((len(data.formal_charge), 4))
    for _ in range((len(data.formal_charge))):
        formal_charge_one_hot[_, :] = one_hot_k_encode(data.formal_charge[_], [-1, 0, 1,'unknown'])
    formal_charge_one_hot = torch.tensor(formal_charge_one_hot).to(torch.float32)

    radical_electrons_one_hot = np.zeros((len(data.radical_electrons), 3))
    for _ in range((len(data.radical_electrons))):
        radical_electrons_one_hot[_, :] = one_hot_k_encode(data.radical_electrons[_], [0, 1, 'unknown'])
    radical_electrons_one_hot = torch.tensor(radical_electrons_one_hot).to(torch.float32)

    hybridization_one_hot = np.zeros((len(data.hybridization), 5))
    for _ in range((len(data.hybridization))):
        hybridization_one_hot[_, :] = one_hot_k_encode(data.hybridization[_], [1, 2, 3, 4, 'unknown'])
    hybridization_one_hot = torch.tensor(hybridization_one_hot).to(torch.float32)

    total_num_Hs_one_hot = np.zeros((len(data.total_num_Hs), 6))
    for _ in range((len(data.total_num_Hs))):
        total_num_Hs_one_hot[_, :] = one_hot_k_encode(data.total_num_Hs[_], [0, 1, 2, 3, 4, 'unknown'])
    total_num_Hs_one_hot = torch.tensor(total_num_Hs_one_hot).to(torch.float32)

    is_in_ring_one_hot = np.zeros((len(data.is_in_ring), 2))
    for _ in range((len(data.is_in_ring))):
        is_in_ring_one_hot[_, :] = one_hot_k_encode(data.is_in_ring[_], [True, False])
    is_in_ring_one_hot = torch.tensor(is_in_ring_one_hot).to(torch.float32)

    is_aromatic_one_hot = np.zeros((len(data.is_aromatic), 2))
    for _ in range((len(data.is_aromatic))):
        is_aromatic_one_hot[_, :] = one_hot_k_encode(data.is_aromatic[_], [True, False])
    is_aromatic_one_hot = torch.tensor(is_aromatic_one_hot).to(torch.float32)

    is_parent_one_hot = np.zeros((len(data.is_parent), 2))
    for _ in range((len(data.is_parent))):
        is_parent_one_hot[_, :] = one_hot_k_encode(data.is_parent[_], [True, False])
    is_parent_one_hot = torch.tensor(is_parent_one_hot).to(torch.float32)
    
    x = torch.cat([atomic_num_one_hot,
                    formal_charge_one_hot,
                    radical_electrons_one_hot,
                    hybridization_one_hot,
                    total_num_Hs_one_hot,
                    is_in_ring_one_hot,
                    is_aromatic_one_hot,
                    is_parent_one_hot],dim=1)
    
    bond_type_one_hot = np.zeros((len(data.bond_type), 5))
    for _ in range((len(data.bond_type))):
        bond_type_one_hot[_, :] = one_hot_k_encode(data.bond_type[_], [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Fragmentation'])
    bond_type_one_hot = torch.tensor(bond_type_one_hot).to(torch.float32)

    edge_attr = bond_type_one_hot
    edge_type = torch.Tensor([0 if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == True else 1 if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == False else 2 for i in range(edge_index.shape[1])]).int()
    edge_index = edge_index
    if type == 'regression1':
        y = torch.tensor(data.intensity_regression1, dtype=torch.float).to(torch.float32)
    if type == 'regression2':
        y = torch.tensor(data.intensity_regression2, dtype=torch.float).to(torch.float32)
    elif type == 'classification':
        y = torch.tensor(data.intensity_classification, dtype=torch.long)
    elif type == 'prediction':
        y = -1
    
    if is_homo:
        homogeneous_data = Data(x=x, edge_attr=edge_attr, edge_type=edge_type, edge_index=edge_index, y=y)
        return homogeneous_data
    else:
        hetero_data = HeteroData()
        hetero_data['parent'].x = x[np.where(data.is_parent.numpy())]
        hetero_data['product'].x = x[np.where(data.is_parent.numpy()==False)]
        hetero_data['parent','bond','parent'].edge_index = edge_index[:,[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == True]]
        hetero_data['product','bond','product'].edge_index = edge_index[:,[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == False]]
        hetero_data['parent','fragment','product'].edge_index = edge_index[:,[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] != data.is_parent.numpy()[edge_index[1,i].numpy().tolist()]]]
        hetero_data['parent','bond','parent'].edge_attr = edge_attr[[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == True]]
        hetero_data['product','bond','product'].edge_attr = edge_attr[[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] == data.is_parent.numpy()[edge_index[1,i].numpy().tolist()] == False]]
        hetero_data['parent','fragment','product'].edge_attr = edge_attr[[i for i in range(edge_index.shape[1]) if data.is_parent.numpy()[edge_index[0,i].numpy().tolist()] != data.is_parent.numpy()[edge_index[1,i].numpy().tolist()]]]
        hetero_data.y = y
        return hetero_data
    

def batch_transfer_pyg_graph(step4FilePath, step5FilePath_regression1, step5FilePath_regression2, step5FilePath_classification, MW_threshold, n_jobs = -1):
    if not step5FilePath_regression1 is None:
        if not os.path.exists(step5FilePath_regression1): os.makedirs(step5FilePath_regression1)
    if not step5FilePath_regression2 is None:
        if not os.path.exists(step5FilePath_regression2): os.makedirs(step5FilePath_regression2)
    if not step5FilePath_classification is None:
        if not os.path.exists(step5FilePath_classification): os.makedirs(step5FilePath_classification)

    step4Files = os.listdir(step4FilePath)

    for i in step4Files:
        print(i)
        ith_Graph_list = pickle_load(step4FilePath + '/' + i)
        ith_Graph_list = [_ for _ in ith_Graph_list if _.graph['product_mz'] >= MW_threshold]

        if not step5FilePath_regression1 is None:
            data_list_regression1 = Parallel(n_jobs = n_jobs, verbose = 0)(delayed(process)(G, type = 'regression1', is_homo = True) for G in tqdm(ith_Graph_list))
            save_file_regression1 = open(step5FilePath_regression1+'/'+i,"wb")
            pickle.dump(data_list_regression1, save_file_regression1)
            save_file_regression1.close()
    
        if not step5FilePath_regression2 is None:
            data_list_regression2 = Parallel(n_jobs = n_jobs, verbose = 0)(delayed(process)(G, type = 'regression2', is_homo = True) for G in tqdm(ith_Graph_list))
            save_file_regression2 = open(step5FilePath_regression2+'/'+i,"wb")
            pickle.dump(data_list_regression2, save_file_regression2)
            save_file_regression2.close()

        if not step5FilePath_classification is None:
            data_list_classification = Parallel(n_jobs = n_jobs, verbose = 0)(delayed(process)(G, type = 'classification', is_homo = True) for G in tqdm(ith_Graph_list))
            save_file_classification = open(step5FilePath_classification+'/'+i,"wb")
            pickle.dump(data_list_classification, save_file_classification)
            save_file_classification.close()

def batch_transfer_pyg_graph_for_new_molecules(step2FilePath, step5FilePath, MW_threshold, n_jobs = -1):
    if not os.path.exists(step5FilePath): os.makedirs(step5FilePath)
    
    step4Files = os.listdir(step2FilePath)

    for i in step4Files:
        print(i)
        ith_Graph_list = pickle_load(step2FilePath + '/' + i)
        ith_Graph_list = [_ for _ in ith_Graph_list if _.graph['product_mz'] >= MW_threshold]

        data_list_regression = Parallel(n_jobs = n_jobs, verbose = 0)(delayed(process)(G, type = 'prediction', is_homo = True) for G in tqdm(ith_Graph_list))
        save_file_regression = open(step5FilePath+'/'+i,"wb")
        pickle.dump(data_list_regression, save_file_regression)
        save_file_regression.close()