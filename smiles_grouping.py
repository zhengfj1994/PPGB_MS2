import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

adduct_dict = {'[M+H]+' : {'mz': 1.00727646677,  'smiles':'.[H+]',     'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'},
               '[H+]' : {'mz': 1.00727646677,  'smiles':'.[H+]',     'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'},
               '[M+Na]+': {'mz': 22.989218, 'smiles':'.[Na+]',    'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'},
               '[Na]+': {'mz': 22.989218, 'smiles':'.[Na+]',    'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'}, 
               '[M+K]+' : {'mz': 38.963158, 'smiles':'.[K+]',     'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'},
               '[K]+' : {'mz': 38.963158, 'smiles':'.[K+]',     'charge_mz': -0.00054857990924, 'charge_smiles': '.[+]', 'ion_mode':'positive'},
               '[M-H]-' : {'mz': -1.00727646677, 'smiles':'.[M-H][-]', 'charge_mz': 0.00054857990924,  'charge_smiles': '.[-]', 'ion_mode':'negative'},
               '[M-H][-]' : {'mz': -1.00727646677, 'smiles':'.[M-H][-]', 'charge_mz': 0.00054857990924,  'charge_smiles': '.[-]', 'ion_mode':'negative'},
               '[+]':{'mz': -0.00054857990924, 'smiles':'.[+]', 'ion_mode':'positive'},
               '[-]':{'mz': 0.00054857990924, 'smiles':'.[-]', 'ion_mode':'negative'}}

def smiles_grouping(fragmentation_as_reaction_list, adduct, mz_threshold):
    ionMode = adduct_dict[adduct]['ion_mode']
    adduct_MW = adduct_dict[adduct]['mz']
    precusor_smiles = fragmentation_as_reaction_list[0]
    product_smiles_list = fragmentation_as_reaction_list[1]
    product_mol_list = [Chem.MolFromSmiles(i) for i in product_smiles_list]
    product_charge_list = [sum([j.GetFormalCharge() for j in i.GetAtoms()]) if i is not None else 99 for i in product_mol_list]

    df = pd.DataFrame({'product_smiles_list':product_smiles_list, 'product_mol_list':product_mol_list, 'product_charge_list':product_charge_list})
    if ionMode == 'positive':
        df_charge = df.loc[(df['product_charge_list'] == 0) | (df['product_charge_list'] == 1)]
    elif ionMode == 'negative':
        df_charge = df.loc[(df['product_charge_list'] == 0) | (df['product_charge_list'] == -1)]
    product_smiles_list = df_charge['product_smiles_list'].tolist()
    product_mol_list = df_charge['product_mol_list'].tolist()
    product_charge_list = df_charge['product_charge_list'].tolist()

    product_MW_list = [round(Descriptors.ExactMolWt(Chem.MolFromSmiles(i))+adduct_MW, 5) if j == 0 else round(Descriptors.ExactMolWt(Chem.MolFromSmiles(i)),5) for i,j in zip(product_smiles_list, product_charge_list)]

    df2 = pd.DataFrame({"smiles":product_smiles_list,"product_mz":product_MW_list}) # Create a dataframe with graph and product mz
    group = df2.groupby("product_mz") # Group the dataframe with product mz

    group_result = [[ithGroup[0],list(ithGroup[1]['smiles']),precusor_smiles] for ithGroup in group]
    
    return [i for i in group_result if i[0] > mz_threshold]


if __name__=='__main__':
    load_file = open('training data/NIST 20/Agilent 6530 Q-TOF/Step 1/AABLHGPVOULICI-ZOFKVTQNSA-N.pkl',"rb") # open file
    fragmentation_as_reaction_list = pickle.load(load_file) # open file
    test = smiles_grouping(fragmentation_as_reaction_list,adduct='[M+H]+')


def batch_smiles_grouping(step1FilePath, step2FilePath, adducts = ['[M+H]+','[M-H]-','[M+Na]+','[M+K]+'], mz_threshold = 25, n_jobs = -1):
    if not os.path.exists(step2FilePath): os.makedirs(step2FilePath)

    for ithAdduct in adducts:
        def my_task(i):
            if ithAdduct + '_' + i not in os.listdir(step2FilePath): # If ith item is not processed
                load_file = open(step1FilePath + '/' + i,"rb") # open file
                try: fragmentation_as_reaction_list = pickle.load(load_file) # open file
                except: return 'error' # if open file failed
                group_result = smiles_grouping(fragmentation_as_reaction_list, adduct = ithAdduct, mz_threshold = mz_threshold) # Merge by group
                save_file = open(step2FilePath + '/' + ithAdduct + '_' + i,"wb") # save file
                pickle.dump(group_result, save_file) # save file
                save_file.close() # save file
                return i
            else:
                return 'done'

        lst1 = os.listdir(step1FilePath)
        Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))