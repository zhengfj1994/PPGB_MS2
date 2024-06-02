import os
import torch
import pickle
import torch_geometric
from utils import pickle_load
from tqdm.notebook import tqdm
from rdkit import Chem, RDLogger
from joblib import Parallel, delayed
from torch_geometric.data import Data



def one_hot_k_encode(x, permitted_list):
    if x not in permitted_list: x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC'
    ]
}

meta_map = {
    'InstrumentManufacturer': ['Sciex', 'Thremo', 'Agilent', 'Bruker', 'Shimadzu', 'Waters', 'Others'],
    'IonSource': ['ESI', 'APCI', 'EI', 'Others'],
    'DetectorType': ['TOF', 'Orbitrap', 'Others'],
    'FragmentationType': ['CID', 'HCD', 'EAD', 'Others'],
    'CollisionGas': ['N2', 'He', 'Others'],
    'AdductType': ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+H-H2O]+', '[M+NH4]+', '[M+FA-H]-', '[2M+H]+', '[2M-H]-', '[M-2H2O+H]+', '[M]+', '[M+Cl]-', '[M-H2O-H]-', '[M+K]+', 'Others']
}

def from_smiles(smiles: str, precusor: bool = True) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    RDLogger.DisableLog('rdApp.*')

    precusor_mol = Chem.MolFromSmiles(smiles, sanitize = True)
    Chem.SanitizeMol(precusor_mol)

    if precusor_mol is None:
        precusor_mol = Chem.MolFromSmiles('')

    xs = []
    for atom in precusor_mol.GetAtoms():
        x = []
        if precusor:
            x += [0]
        else:
            x += [1]
        x.append(atom.GetAtomMapNum())
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x += one_hot_k_encode(atom.GetChiralTag(),x_map['chirality'])
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x += one_hot_k_encode(atom.GetHybridization(),x_map['hybridization'])
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.float).view(-1,26)

    edge_indices, edge_attrs, edge_type = [], [], []
    for bond in precusor_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
        if precusor:
            edge_type += [0, 0]
        else:
            edge_type += [1, 1]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_attr=edge_attr, edge_type=edge_type, edge_index=edge_index)


def precusor_product_graph_combination(precusor_graph, product_graphs):
    fragmentation_edge_attr = torch.Tensor([4]).view(-1, 1)
    combined_graph = precusor_graph
    for ith_product_graph in product_graphs:
        combined_graph = Data(x = torch.cat([combined_graph['x'], ith_product_graph['x']]),
                              edge_index = torch.cat([combined_graph['edge_index'], ith_product_graph['edge_index']+combined_graph['x'].shape[0]],dim=1),
                              edge_attr = torch.cat([combined_graph['edge_attr'], ith_product_graph['edge_attr']]),
                              edge_type = combined_graph['edge_type']+ith_product_graph['edge_type'],)
    isPrecusor = combined_graph['x'][:,0].numpy().tolist()
    atom_map_num = combined_graph['x'][:,1].numpy().tolist()
    raw_edge_index = combined_graph['edge_index']
    raw_edge_attr = combined_graph['edge_attr']
    raw_edge_type = combined_graph['edge_type']
    for ith_isPrecusor, ith_atom_map_num in zip(isPrecusor, atom_map_num):
        if ith_isPrecusor == 1: break
        precusor_product_map = [i for i, x in enumerate(atom_map_num) if x == ith_atom_map_num]
        if len(precusor_product_map) == 1:
            continue
        else:
            for ith_pair in range(1, len(precusor_product_map)):
                temp_edge_index = torch.Tensor([[precusor_product_map[ith_pair]], [precusor_product_map[0]]]).to(torch.long)
                raw_edge_index = torch.cat((raw_edge_index, temp_edge_index), dim=1)
                raw_edge_attr = torch.cat((raw_edge_attr, fragmentation_edge_attr), dim=0)
                raw_edge_type += [2]
    combined_graph['x'] = torch.cat((combined_graph['x'][:, 2:],combined_graph['x'][:, :1]), dim=1)
    combined_graph['edge_attr'] = raw_edge_attr
    combined_graph['edge_type'] = torch.Tensor(raw_edge_type).long()
    combined_graph['edge_index'] = raw_edge_index.view(2, -1)
    return combined_graph


def precusor_product_graphs_generation(raw_data, type):
    precusor_smiles = raw_data[2]
    product_smiles = raw_data[1]
    precusor_graph = from_smiles(smiles=precusor_smiles, precusor=True)
    product_graphs = [from_smiles(smiles=i, precusor=False) for i in product_smiles]
    precusor_product_graph = precusor_product_graph_combination(precusor_graph=precusor_graph, product_graphs=product_graphs)

    if type == 'prediction':
        precusor_product_graph['mz'] = torch.tensor(raw_data[0], dtype=torch.float).to(torch.float32)
    else:
        precusor_product_graph['weight'] = torch.tensor(raw_data[3]['weight'], dtype=torch.float).to(torch.float32)

    precusor_product_graph['y'] = torch.tensor(raw_data[3]['intensity'], dtype=torch.float).to(torch.float32)
    
    InstrumentManufacturer = torch.tensor(one_hot_k_encode(raw_data[4]['InstrumentManufacturer'],meta_map['InstrumentManufacturer']))
    IonSource = torch.tensor(one_hot_k_encode(raw_data[4]['IonSource'],meta_map['IonSource']))
    DetectorType = torch.tensor(one_hot_k_encode(raw_data[4]['DetectorType'],meta_map['DetectorType']))
    FragmentationType = torch.tensor(one_hot_k_encode(raw_data[4]['FragmentationType'],meta_map['FragmentationType']))
    CollisionGas = torch.tensor(one_hot_k_encode(raw_data[4]['CollisionGas'],meta_map['CollisionGas']))
    AdductType = torch.tensor(one_hot_k_encode(raw_data[4]['AdductType'],meta_map['AdductType']))
    CE = torch.unsqueeze(torch.tensor(raw_data[4]['CE'], dtype=torch.float).to(torch.float32),dim=0)
    KE = torch.unsqueeze(torch.tensor(raw_data[4]['KE'], dtype=torch.float).to(torch.float32),dim=0)
    EBC = torch.unsqueeze(torch.tensor(raw_data[4]['EBC'], dtype=torch.float).to(torch.float32),dim=0)
    
    precusor_product_graph['MetaData'] = torch.cat((InstrumentManufacturer,IonSource,DetectorType,FragmentationType,CollisionGas,AdductType,CE,KE,EBC), dim=0).unsqueeze(0)


    return precusor_product_graph


def batch_transfer_pyg_graph(step4FilePath, step5FilePath, MW_threshold, n_jobs = -1):
    if not os.path.exists(step5FilePath): os.makedirs(step5FilePath)

    step4Files = os.listdir(step4FilePath)

    def my_task(i):
        ith_Graph_list = pickle_load(step4FilePath + '/' + i)
        ith_Graph_list['group_result'] = [_ for _ in ith_Graph_list if _[0] >= MW_threshold]
        
        if i not in os.listdir(step5FilePath):
            data_list = precusor_product_graphs_generation(ith_Graph_list, type = 'regression')
            save_file = open(step5FilePath+'/'+i,"wb")
            pickle.dump(data_list, save_file)
            save_file.close()

    lst1 = step4Files
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))