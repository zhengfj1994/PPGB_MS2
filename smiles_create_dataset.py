import os
import torch
import math
import pickle
import random
import functools
import multiprocessing
import numpy as np
import os.path as osp
from torch import Tensor
from joblib import Parallel, delayed
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
from utils import pickle_load
from utils import raw_LDS_weight
from smiles_to_pyg_graph import precusor_product_graphs_generation
import dill
from joblib import Parallel, delayed
from tqdm import tqdm

def create_all_smiles_list(step4FilePath,step5FilePath,int_threshold,bin_size,weighted_power,batch_size,random_seed):
    random.seed(random_seed)
    step5FilePath_raw = step5FilePath+'/'+'raw'
    if not os.path.exists(step5FilePath_raw): os.makedirs(step5FilePath_raw)

    step4_list = [pickle_load(step4FilePath + '/' + i) for i in os.listdir(step4FilePath)]
    step4_list = [i for k in step4_list for i in k]
    step4_list = [_ for _ in step4_list if _[3]['intensity'] > int_threshold]
    step4_list = raw_LDS_weight(step4_list, bin_size=bin_size, weighted_power=weighted_power)
    random.shuffle(step4_list)
    num_groups = len(step4_list) // batch_size
    for i in tqdm(range(num_groups), desc="saving"):
        group = step4_list[i*batch_size : (i+1)*batch_size] 
        group_filename = f"{step5FilePath_raw}/raw_{i}.pkl"
        if not osp.exists(group_filename):
            with open(group_filename, "wb") as f:
                pickle.dump(group, f)
    if len(step4_list) % batch_size != 0:
        last_group = step4_list[num_groups*batch_size:]
        last_group_filename = f"{step5FilePath_raw}/raw_{num_groups}.pkl"
        if not osp.exists(group_filename):
            with open(last_group_filename, "wb") as f:
                pickle.dump(last_group, f)


def list_collate_fn(batch):
    data_list = []
    for ith_list in batch:
        for data in ith_list:
            data_list.append(data)
    return Batch.from_data_list(data_list)


class mydataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None, batch_size = 256, type = 'regression', n_jobs = 12):
        self.batch_size = batch_size
        self.type = type
        self.n_jobs = n_jobs
        super(mydataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)[0]
    
    @property
    def processed_file_names(self):
        self.data = pickle_load(self.raw_dir + '/' + os.listdir(self.raw_dir)[0])
        return [f'data_{i}.pt' for i in range(len(os.listdir(self.raw_dir)))]

    def download(self):
        pass

    def process(self):
        lst1 = os.listdir(self.raw_dir)
        Parallel(n_jobs = self.n_jobs, verbose = 1)(delayed(self.process_batch)(i) for i in tqdm(lst1))

    def process_batch(self, ith_raw_file):
        ith_raw_data = pickle_load(f'{self.raw_dir}/{ith_raw_file}')
        file_path = os.path.join(self.processed_dir, f'data_{ith_raw_file.split("_")[-1].split(".")[0]}.pt')
        if not osp.exists(file_path):
            merged_data = None
            for j in ith_raw_data:
                data = precusor_product_graphs_generation(j, type = self.type)
                if merged_data is None:
                    merged_data = [data]
                else:
                    merged_data += [data]
            torch.save(merged_data, file_path)

    def len(self):
        return len(os.listdir(self.raw_dir))

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    @property
    def feature_size(self):
        return self[0].x.shape[1]

    @property
    def edge_dim(self):
        return self[0].edge_attr.shape[1]