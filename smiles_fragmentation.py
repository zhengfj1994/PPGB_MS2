import os
import pickle
import pandas as pd
from rdkit import Chem
from itertools import chain
from tqdm.notebook import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed

########################################## fragmentation reaction
crf1_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1][*+0:2]>>([*+0:0]=[*+0:1].[*+0:2])')
crf1_1 = AllChem.ReactionFromSmarts('[*+0;!H0:0]=[*+0:1][*+0:2]>>([*+0:0]#[*+0:1].[*+0:2])')
crf2_0 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1][*!H0+0:2]>>([*+0:0].[*+0:1]=[*+0:2])')
crf2_1 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1]=[*!H0+0:2]>>([*+0:0].[*+0:1]#[*+0:2])')
crf3_0 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf3_1 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]#[*+0:5])')
crf4_0 = AllChem.ReactionFromSmarts('[*+0:0]=[*+0:1][*+0:2][*+0:3][*+0;!H0:4]>>([*+0:0][*+0:1]=[*+0:2].[*+0:3]=[*+0:4])')
crf5_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1][*+0:2][*+0;!H0:3]>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3])')
crf6_0 = AllChem.ReactionFromSmarts('[*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6]=[*+0:7]>>([*+0:0]=[*+0:7].[*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6]=[*+0:1])')
crf7_0 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf7_1 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf7_2 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]=[*+0:5])')
crf7_3 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]#[*+0:5])')
crf8_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2]>>([*+0:0]=[*+0:1][*+0:2])')
crf8_1 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4])')
crf8_2 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4][*+0:5]=[*+0:6]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6])')
crf9_0 = AllChem.ReactionFromSmarts('[*:0]@[C:1](=[O:2])@[*:3]>>([*:0][*:3].[C-:1]#[O+:2])')
crf11_0= AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0;!H0:1]>>([*+0:0]=[*+0:1])')
crf11_1= AllChem.ReactionFromSmarts('[*+0;!H0:0]=[*+0;!H0:1]>>([*+0:0]#[*+0:1])')
crf12_0= AllChem.ReactionFromSmarts('[*+0:0][*+0:1][*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]=[*+0:2])')
crf12_1= AllChem.ReactionFromSmarts('[*+0:0][*+0:1]=[*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]#[*+0:2])')
crf13_0= AllChem.ReactionFromSmarts('[*+0:0][C+0:1](=[O+0:2])[OH+0:3]>>([*+0:0].[O+0:2]=[C+0:1]=[O+0:3])')
crf13_1= AllChem.ReactionFromSmarts('[C+0;H1:0](=[O+0:1])[O+0:2][*+0:3]>>([O+0:1]=[C+0:0]=[O+0:2].[*+0:3])')
crf13_2= AllChem.ReactionFromSmarts('[*+0:0][C+0:1](=[O+0:2])[O+0:3][*+0:4]>>([O+0:2]=[C+0:1]=[O+0:3].[*+0:0][*+0:4])')
cmf2_0 = AllChem.ReactionFromSmarts('[N+0,O+0,S+0:0][*+0:1][*:2]>>([N+1,O+1,S+1:0]=[*+0:1].[*+0:2])')
cmf2_1 = AllChem.ReactionFromSmarts('[N+0,O+0,S+0:0]=[*+0:1][*:2]>>([N+1,O+1,S+1:0]#[*+0:1].[*+0:2])')
cmf7_0 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1]>>([*+1:0].[*-1:1])')
cmf13_0= AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+1:4]1.[*:5])')
########################################## fragmentation reaction

# Create a list to save reactions
reactions = [crf1_0,crf1_1,crf2_0,crf2_1,crf3_0,crf3_1,crf4_0,crf5_0,crf6_0,crf7_0,crf7_1,crf7_2,crf7_3,crf8_0,crf8_1,crf8_2,crf9_0,
             crf11_0,crf11_1,crf12_0,crf12_1,crf13_0,crf13_1,crf13_2,cmf2_0,cmf2_1,cmf7_0,cmf13_0]

def remove_extra_Hs(mol):
    for atom in mol.GetAtoms():
        bonds = atom.GetBonds() # Get all bonds of an atom
        bonds_prop = [ithBond.GetBondTypeAsDouble() for ithBond in bonds] # Get all bond type as a list
        defaultValence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetSymbol()) # Get the default valence of atom
        numExplicitHs = atom.GetNumExplicitHs() # Get the number of explicit Hs
        numFormalCharge = atom.GetFormalCharge() # Get formal charge

        actualValence = int(sum(bonds_prop) + numExplicitHs - numFormalCharge) # use bonds, explicit Hs, formal charge to calculate actual valence
        if actualValence > defaultValence:
            if numExplicitHs-(actualValence-defaultValence) >= 0:
                atom.SetNumExplicitHs(numExplicitHs-(actualValence-defaultValence))
    return mol


def make_rxns(rxn, precusor_smiles):
    precusor_mol = Chem.MolFromSmiles(precusor_smiles) # get precusor mol
    Chem.Kekulize(precusor_mol, clearAromaticFlags=True) # kekulize molecule
    for atom in precusor_mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber') != 1:
            atom.SetIntProp('molAtomMapNumber', atom.GetIdx())
        atom.SetIntProp('originalAtomMapNumber', atom.GetAtomMapNum())

    product_set1 = rxn.RunReactants([precusor_mol])
    
    precusor_precusor_reaction = AllChem.ChemicalReaction()
    precusor_precusor_reaction.AddReactantTemplate(precusor_mol)
    precusor_precusor_reaction.AddProductTemplate(precusor_mol)
    precusor_precusor_reaction_smiles = AllChem.ReactionToSmiles(precusor_precusor_reaction)

    if len(product_set1) == 0: return [precusor_precusor_reaction_smiles]

    atomIdxs = [atom.GetIdx() for atom in precusor_mol.GetAtoms()]
    atomMaps = [int(atom.GetProp('originalAtomMapNumber')) for atom in precusor_mol.GetAtoms()]
    atomIdxs_Maps = dict(zip(atomIdxs, atomMaps))

    new_rxns = []

    for pset in product_set1:
        for prod in pset:
            prod = remove_extra_Hs(prod)
            new_rxn = AllChem.ChemicalReaction()
            new_rxn.AddReactantTemplate(precusor_mol)

            for a in prod.GetAtoms():
                if not a.GetSymbol() == '*':
                    a.SetIntProp('molAtomMapNumber', atomIdxs_Maps[int(a.GetProp('react_atom_idx'))])
                    a.SetIntProp('originalAtomMapNumber', atomIdxs_Maps[int(a.GetProp('react_atom_idx'))])
            
            new_rxn.AddProductTemplate(prod)
            new_rxn = AllChem.ReactionToSmiles(new_rxn)
            new_rxns.append(new_rxn)
    return [precusor_precusor_reaction_smiles] + new_rxns


def batch_fragmentation(precusor_smiles, reactions):
    reaction_smiles = [make_rxns(rxn = ithReaction, precusor_smiles = precusor_smiles) for ithReaction in reactions]
    result = list({*chain.from_iterable(i for i in reaction_smiles)})
    return result


def two_steps_fragmentation(precusor_smiles, reactions):
    print(precusor_smiles)
    reaction_smiles_step1 = batch_fragmentation(precusor_smiles = precusor_smiles, reactions = reactions)

    if len(reaction_smiles_step1) > 0:
        product_smiles_step1 = [i.split('>>')[1][1:-1] if '.' in i else i.split('>>')[1] for i in reaction_smiles_step1]
        product_smiles_step1 = [i.split('.') for i in product_smiles_step1]
        product_smiles_step1 = [*chain.from_iterable(i for i in product_smiles_step1)]
        product_mol_step1 = [Chem.MolFromSmiles(i) for i in product_smiles_step1]
        product_inchikey_step1 = [Chem.MolToInchiKey(i) if i is not None else None for i in product_mol_step1]
        step1_df = pd.DataFrame({'smiles':product_smiles_step1, 
                        'inchikey':product_inchikey_step1})
        step1_df.dropna(subset=['smiles', 'inchikey'], inplace=True)
        unique_step1_df = step1_df.drop_duplicates(subset=['inchikey'])
        unique_product_smiles_step1 = unique_step1_df['smiles'].tolist()

        reaction_smiles_step2 = [batch_fragmentation(precusor_smiles = i, reactions = reactions) for i in unique_product_smiles_step1]
        reaction_smiles_step2 = [*chain.from_iterable(x for x in reaction_smiles_step2)]
        if len(reaction_smiles_step2) > 0:
            product_smiles_step2 = [i.split('>>')[1][1:-1] if '.' in i else i.split('>>')[1] for i in reaction_smiles_step2]
            product_smiles_step2 = [i.split('.') for i in product_smiles_step2]
            product_smiles_step2 = [*chain.from_iterable(i for i in product_smiles_step2)]
            mol_step2 = [Chem.MolFromSmiles(i) for i in product_smiles_step2]
            atomMap = [tuple(sorted([atom.GetAtomMapNum() for atom in i.GetAtoms()])) if i is not None else tuple([]) for i in mol_step2]
            ExactMolWt = [Descriptors.ExactMolWt(i) if i is not None else -100 for i in mol_step2]
            NumRadicalElectrons = [Descriptors.NumRadicalElectrons(i) if i is not None else 0 for i in mol_step2]
            NumValenceElectrons = [Descriptors.NumValenceElectrons(i) if i is not None else 0 for i in mol_step2]
            step2_df = pd.DataFrame({'atomMap':atomMap, 
                            'ExactMolWt':ExactMolWt, 
                            'NumRadicalElectrons':NumRadicalElectrons, 
                            'NumValenceElectrons':NumValenceElectrons, 
                            'product_smiles_step2': product_smiles_step2})
            unique_step2_df = step2_df.drop_duplicates(subset=['atomMap', 'ExactMolWt', 'NumRadicalElectrons', 'NumValenceElectrons'])
            unique_product_smiles_step2 = unique_step2_df['product_smiles_step2'].tolist()
            return reaction_smiles_step1[0].split('>>')[0], [reaction_smiles_step1[0].split('>>')[0]] + unique_product_smiles_step2
    else:
        return reaction_smiles_step1[0].split('>>')[0], [reaction_smiles_step1[0].split('>>')[0]]
    

def three_steps_fragmentation(precusor_smiles, reactions):

    reaction_smiles_step1 = batch_fragmentation(precusor_smiles = precusor_smiles, reactions = reactions)
    
    product_smiles_step1 = [i.split('>>')[1][1:-1] if '.' in i else i.split('>>')[1] for i in reaction_smiles_step1]
    product_smiles_step1 = [i.split('.') for i in product_smiles_step1]
    product_smiles_step1 = [*chain.from_iterable(i for i in product_smiles_step1)]
    product_smiles_step1 = list(set(product_smiles_step1))

    reaction_smiles_step2 = [batch_fragmentation(precusor_smiles = i, reactions = reactions) for i in product_smiles_step1]
    reaction_smiles_step2 = [*chain.from_iterable(x for x in reaction_smiles_step2)]
    product_smiles_step2 = [i.split('>>')[1][1:-1] if '.' in i else i.split('>>')[1] for i in reaction_smiles_step2]
    product_smiles_step2 = [i.split('.') for i in product_smiles_step2]
    product_smiles_step2 = [*chain.from_iterable(i for i in product_smiles_step2)]
    product_smiles_step2 = list(set(product_smiles_step2))

    reaction_smiles_step3 = [batch_fragmentation(precusor_smiles = i, reactions = reactions) for i in product_smiles_step2]
    reaction_smiles_step3 = [*chain.from_iterable(x for x in reaction_smiles_step3)]
    product_smiles_step3 = [i.split('>>')[1][1:-1] if '.' in i else i.split('>>')[1] for i in reaction_smiles_step3]
    product_smiles_step3 = [i.split('.') for i in product_smiles_step3]
    product_smiles_step3 = [*chain.from_iterable(i for i in product_smiles_step3)]
    product_smiles_step3 = list(set(product_smiles_step3))

    mol_final = [Chem.MolFromSmiles(i) for i in product_smiles_step3]
    atomMap_final = [tuple(sorted([atom.GetAtomMapNum() for atom in i.GetAtoms()])) if i is not None else tuple([]) for i in mol_final]
    ExactMolWt_final = [Descriptors.ExactMolWt(i) if i is not None else -100 for i in mol_final]
    NumRadicalElectrons_final = [Descriptors.NumRadicalElectrons(i) if i is not None else 0 for i in mol_final]
    NumValenceElectrons_final = [Descriptors.NumValenceElectrons(i) if i is not None else 0 for i in mol_final]
    df_final = pd.DataFrame({'atomMap':atomMap_final, 
                    'ExactMolWt':ExactMolWt_final, 
                    'NumRadicalElectrons':NumRadicalElectrons_final, 
                    'NumValenceElectrons':NumValenceElectrons_final, 
                    'product_smiles': product_smiles_step3})
    unique_df_final = df_final.drop_duplicates(subset=['atomMap', 'ExactMolWt', 'NumRadicalElectrons', 'NumValenceElectrons'])
    unique_product_smiles_final = unique_df_final['product_smiles'].tolist()
            
    return reaction_smiles_step1[0].split('>>')[0], unique_product_smiles_final


def batch_smiles_fragmentation(rawCsvFilePath, step1FilePath, reactions, steps, n_jobs = -1):
    if not os.path.exists(step1FilePath): os.makedirs(step1FilePath)
    rawCsvData = pd.read_csv(rawCsvFilePath, encoding=u'gbk')
    rawCsvData.drop_duplicates(subset=['inchikey'], keep='first', inplace=True)
    a = rawCsvData['inchikey'].values.tolist()
    b = [_.replace('.pkl','') for _ in os.listdir(step1FilePath)]
    c1 = list(set(a)- set(b))
    csvData = rawCsvData[rawCsvData['inchikey'].isin(c1)]

    if steps == 2:
        def my_task(i):
            inchikey_file_name = str(csvData['inchikey'].values[i]) + '.pkl'
            if inchikey_file_name not in os.listdir(step1FilePath): # If ith item is not processed
                mol = Chem.MolFromSmiles(csvData['smiles'].values[i])
                if mol == None:
                    return 'none'
                if Chem.rdmolops.GetFormalCharge(mol) != 0:
                    return 'none'
                if Descriptors.ExactMolWt(mol) > 1000:
                    return '> 1000 Da'
                precusor_maped_smiles, product_maped_smiles = two_steps_fragmentation(precusor_smiles = csvData['smiles'].values[i], reactions=reactions)

                save_file = open(step1FilePath + '/' + str(csvData['inchikey'].values[i]) + '.pkl',"wb")
                pickle.dump([precusor_maped_smiles,product_maped_smiles], save_file)
                save_file.close()
                return i
            else:
                return 'done'
    elif steps == 3:
        def my_task(i):
            inchikey_file_name = str(csvData['inchikey'].values[i]) + '.pkl'
            if inchikey_file_name not in os.listdir(step1FilePath): # If ith item is not processed
                mol = Chem.MolFromSmiles(csvData['smiles'].values[i])
                if mol == None:
                    return 'none'
                if Chem.rdmolops.GetFormalCharge(mol) != 0:
                    return 'none'
                if Descriptors.ExactMolWt(mol) > 1000:
                    return '> 1000 Da'
                precusor_maped_smiles, product_maped_smiles = three_steps_fragmentation(precusor_smiles = csvData['smiles'].values[i], reactions=reactions)

                save_file = open(step1FilePath + '/' + str(csvData['inchikey'].values[i]) + '.pkl',"wb")
                pickle.dump([precusor_maped_smiles,product_maped_smiles], save_file) #顺序存入变量
                save_file.close()
                return i
            else:
                return 'done'
    else:
        return('error')

    lst1 = list(range(len(csvData)))
    Parallel(n_jobs = n_jobs, verbose = 1)(delayed(my_task)(i) for i in tqdm(lst1))