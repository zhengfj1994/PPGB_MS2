import copy
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors


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


def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atom_map_num = atom.GetAtomMapNum(),
                   atomic_num = atom.GetAtomicNum(),
                   formal_charge = atom.GetFormalCharge(),
                   radical_electrons = atom.GetNumRadicalElectrons(),
                   hybridization = atom.GetHybridization(),
                   total_num_Hs = atom.GetTotalNumHs(),
                   total_degree = atom.GetTotalDegree(),
                   mass = atom.GetMass(),
                   is_in_ring = atom.IsInRing(),
                   is_aromatic = atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type = bond.GetBondType())
    return G


def nx_to_mol(G):
    mol = Chem.RWMol()
    atom_map_num = nx.get_node_attributes(G, 'atom_map_num')
    atomic_num = nx.get_node_attributes(G, 'atomic_num')
    formal_charge = nx.get_node_attributes(G, 'formal_charge')
    radical_electrons = nx.get_node_attributes(G, 'radical_electrons')
    hybridization = nx.get_node_attributes(G, 'hybridization')
    total_num_Hs = nx.get_node_attributes(G, 'total_num_Hs')
    total_degree = nx.get_node_attributes(G, 'total_degree')
    is_in_ring = nx.get_node_attributes(G, 'is_in_ring')
    is_aromatic = nx.get_node_attributes(G, 'is_aromatic')

    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_num[node])
        a.SetAtomMapNum(atom_map_num[node])
        a.SetProp('atomNote', str(atom_map_num[node]))
        a.SetFormalCharge(formal_charge[node])
        a.SetNumRadicalElectrons(radical_electrons[node])
        a.SetHybridization(hybridization[node])
        a.SetIsAromatic(is_aromatic[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)
    
    mol.UpdatePropertyCache()
    
    return mol


def kekule_graph(graph):
    mol = nx_to_mol(graph)
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol)
    charge_num = Chem.rdmolops.GetFormalCharge(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    NumFormalCharges = sum([abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()])
    num_h = []
    for u in range(mol.GetNumAtoms()):
        num_h += [mol.GetAtomWithIdx(u).GetTotalNumHs()]
    suppl = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL | Chem.ALLOW_CHARGE_SEPARATION | Chem.ALLOW_INCOMPLETE_OCTETS)

    kekule_graphs = []
    for ithMol in suppl:
        if ithMol == None:
            continue
        if not Descriptors.NumValenceElectrons(ithMol) == NumValenceElectrons:
            continue
        if not sum([abs(atom.GetFormalCharge()) for atom in ithMol.GetAtoms()]) == NumFormalCharges:
            continue
        
        ith_num_h = []
        for u in range(ithMol.GetNumAtoms()):
            ith_num_h += [ithMol.GetAtomWithIdx(u).GetTotalNumHs()]
        if not ith_num_h == num_h:
            continue
        for ithAtom in ithMol.GetAtoms():
            ithAtom.SetIsAromatic(False)
        ithGraph = mol_to_nx(ithMol)
        for ithNode in ithGraph.nodes():
            ithGraph.add_node(ithNode, is_parent=graph.nodes[list(graph.nodes())[0]]['is_parent'])
        ithGraph.graph = graph.graph
        kekule_graphs += [ithGraph]
    if len(kekule_graphs) == 0:
        return [graph]
    else:
        return kekule_graphs


def sanitize_graph(graph):
    for ithNode in graph.nodes():
        graph.nodes[ithNode]['is_aromatic'] = False
    mol = nx_to_mol(graph)
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol)
    sanitized_graph = mol_to_nx(mol)
    for ithNode in sanitized_graph.nodes():
        sanitized_graph.add_node(ithNode, is_parent=graph.nodes[list(graph.nodes())[0]]['is_parent'])
    sanitized_graph.graph = graph.graph
    return(sanitized_graph)


def graph_duplicate(fragment_result_list, step = 1):
    if len(fragment_result_list) > 1:
        if step == 1:
            product_SMILES = [fragment_result.graph['product_SMILES'] for fragment_result in fragment_result_list]
            parent_mz = [fragment_result.graph['parent_mz'] for fragment_result in fragment_result_list]
            info_df = pd.DataFrame({'product_SMILES':product_SMILES, 'parent_mz':parent_mz})
            index = sorted(info_df.groupby(['product_SMILES'])['parent_mz'].idxmax())
        else:
            MW = [round(fragment_result.graph['product_mz'],5) for fragment_result in fragment_result_list]
            atom_map_num = [",".join([str(x) for x in sorted(list(nx.get_node_attributes(fragment_result, 'atom_map_num').values()))]) for fragment_result in fragment_result_list] # 列表推导式获取各种信息
            parent_mz = [fragment_result.graph['parent_mz'] for fragment_result in fragment_result_list]
            info_df = pd.DataFrame({'MW':MW, 'atom_map_num':atom_map_num, 'parent_mz':parent_mz})
            index = sorted(info_df.groupby(['MW','atom_map_num'])['parent_mz'].idxmax())
        fragment_result_list = [fragment_result_list[i] for i in index]
    return(fragment_result_list)


def preprocess_raw_graph(raw_graph):
    raw_graph_product_SMILES = raw_graph.graph['product_SMILES']
    splited_raw_graph_product_SMILES = raw_graph_product_SMILES.split('.')
    if len(splited_raw_graph_product_SMILES) == 1:
        charge_position = True
        charge_source = []
    elif len(splited_raw_graph_product_SMILES) == 2:
        charge_position = False
        charge_source = splited_raw_graph_product_SMILES[1]
    else:
        charge_position = 'wrong'
    parent_graph = copy.deepcopy(raw_graph)
    [parent_graph.remove_node(i) for i in range(int(parent_graph.number_of_nodes())) if not parent_graph.nodes[i]['is_parent']]
    product_graph = copy.deepcopy(raw_graph)
    [product_graph.remove_node(i) for i in range(int(product_graph.number_of_nodes())) if product_graph.nodes[i]['is_parent']]
    product_graph = nx.disjoint_union(product_graph,nx.Graph())
    kekuled_product_graphs = kekule_graph(product_graph)
    return(charge_position, charge_source, parent_graph, kekuled_product_graphs)


def combind_parent_product_graph(raw_graph, parent_graph, ith_productGraph, fragmentation, adduct, charge_source, charge_position):
    ith_subGraph = nx.disjoint_union(parent_graph, ith_productGraph)
    atom_map_num = [ith_subGraph.nodes[i]['atom_map_num'] for i in range(ith_subGraph.number_of_nodes())]
    unique_atom_map_num = [i for i in list(set(atom_map_num)) if atom_map_num.count(i) == 2]
    atom_map_bond = [[id for id, value in enumerate(atom_map_num) if value == i] for i in unique_atom_map_num]
    [ith_subGraph.add_edges_from([(i[0], i[1], {'bond_type':'Fragmentation'})]) for i in atom_map_bond]
    ith_subGraph.graph['fragment_type'] = raw_graph.graph['fragment_type'] + '+' + fragmentation
    ith_subGraph.graph['adduct'] = adduct
    ith_subGraph.graph['parent_SMILES'] = raw_graph.graph['product_SMILES']
    ith_subGraph.graph['parent_mz'] = raw_graph.graph['product_mz']
    if charge_position:
        ith_subGraph.graph['product_SMILES'] = Chem.MolToSmiles(nx_to_mol(ith_productGraph))
        ith_subGraph.graph['product_mz'] = Descriptors.ExactMolWt(nx_to_mol(ith_productGraph))
    else:
        ith_subGraph.graph['product_SMILES'] = Chem.MolToSmiles(nx_to_mol(ith_productGraph)) + '.' + charge_source
        ith_subGraph.graph['product_mz'] = Descriptors.ExactMolWt(nx_to_mol(ith_productGraph)) + adduct_dict[charge_source]['mz']
    return(ith_subGraph)


def raw_graph_generator(smiles,adduct):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    charge_num = Chem.rdmolops.GetFormalCharge(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    parent_graph = mol_to_nx(mol)
    [parent_graph.add_node(i, is_parent = True) for i in range(parent_graph.number_of_nodes())]
    product_graph = mol_to_nx(mol)
    [product_graph.add_node(i, is_parent = False) for i in range(product_graph.number_of_nodes())]
    raw_graph = nx.disjoint_union(parent_graph,product_graph)
    [raw_graph.add_edges_from([(i, i+parent_graph.number_of_nodes(), {'bond_type':'Fragmentation'})]) for i in range(product_graph.number_of_nodes())]
    raw_graph.graph['fragment_type'] = 'parent_ion'
    raw_graph.graph['adduct'] = adduct
    if charge_num == 0:
        raw_graph.graph['parent_SMILES'] = Chem.MolToSmiles(mol) + adduct_dict[adduct]['smiles']
        raw_graph.graph['parent_mz'] = Descriptors.ExactMolWt(mol) + adduct_dict[adduct]['mz']
        raw_graph.graph['product_SMILES'] = Chem.MolToSmiles(mol) + adduct_dict[adduct]['smiles']
        raw_graph.graph['product_mz'] = Descriptors.ExactMolWt(mol) + adduct_dict[adduct]['mz']
    else:
        raw_graph.graph['parent_SMILES'] = Chem.MolToSmiles(mol)
        raw_graph.graph['parent_mz'] = Descriptors.ExactMolWt(mol)
        raw_graph.graph['product_SMILES'] = Chem.MolToSmiles(mol)
        raw_graph.graph['product_mz'] = Descriptors.ExactMolWt(mol)
    return(raw_graph)