import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDLogger
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors


filename = 'tested_molecules-1.csv'
esol_data = pd.read_csv(filename)
molecules = []
for i in range(len(esol_data['SMILES'])):
    molecule = Chem.MolFromSmiles(esol_data['SMILES'][i])
    molecules.append(molecule)

def getMolDescriptors(mol,missingVal=None):
    ''' calculate the full list of descriptors for a molecule

        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res
d = {}
for k in range(len(molecules)):
    descriptors = getMolDescriptors(molecules[k])
    d[molecules[k]] = descriptors
