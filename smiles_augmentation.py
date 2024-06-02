import os
import copy
import math
import random
import pickle
import numpy as np
import networkx as nx
from itertools import product
from itertools import combinations
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

def smiles_augmentation(raw_smiles, repeat, augmentation_ratio, max_augmentation_times):
    max_augmentation_times = int(max_augmentation_times)
    random.seed(99)
    fragments = raw_smiles[1]
    repeat = max(len(fragments)-repeat,math.ceil(len(fragments)*augmentation_ratio))
    sub_fragments = []
    for i in range(repeat,len(fragments)):
        sub_fragments += list(combinations(fragments, i))
    if len(sub_fragments) > max_augmentation_times:
        sub_fragments = random.sample(sub_fragments, max_augmentation_times)
    sub_fragments += [tuple(fragments)]

    augmented_smiles = []
    if len(raw_smiles) == 5:
        for i in sub_fragments:
            augmented_smiles += [[raw_smiles[0]] + [i] + [raw_smiles[2]] + [raw_smiles[3]] + [raw_smiles[4]]]
    else:
        for i in sub_fragments:
            augmented_smiles += [[raw_smiles[0]] + [i] + [raw_smiles[2]] + [raw_smiles[3]]]
    return augmented_smiles

def batch_smiles_augmentation(step3FilePath, step4FilePath, repeat_1, augmentation_ratio_1, max_augmentation_times_1, repeat_2, augmentation_ratio_2, max_augmentation_times_2, int_threshold, n_jobs = -1):
    if not os.path.exists(step4FilePath): os.makedirs(step4FilePath)

    def my_task(i):
        load_file = open(step3FilePath + '/' + i,"rb") # open file
        try: fragment_list = pickle.load(load_file) # open file
        except: return 'error' # if open file failed
        augmented_fragment_list = [smiles_augmentation(raw_smiles=_, repeat=repeat_1, augmentation_ratio=augmentation_ratio_1, max_augmentation_times=max_augmentation_times_1) \
                                   if _[3]['intensity']>int_threshold else \
                                   smiles_augmentation(raw_smiles=_, repeat=repeat_2, augmentation_ratio=augmentation_ratio_2, max_augmentation_times=max_augmentation_times_2) \
                                   for _ in fragment_list] # Data augmentation
        augmented_fragment_list = [i for k in augmented_fragment_list for i in k]
        save_file = open(step4FilePath + '/' + i,"wb") # save file
        pickle.dump(augmented_fragment_list, save_file) # save file
        save_file.close() # save file
        return i
    
    lst1 = os.listdir(step3FilePath)
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))