import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
def open_excel_filter(filepath):
    # Specify the path to the CSV file
    csv_file_path = r"C:\Users\20212807\OneDrive - TU Eindhoven\Advanced programming\Group Assignment\tested_molecules-1.csv"
    # Read the CSV file
    data_frame = pd.read_csv(csv_file_path)
    # Filter the rows where the right column value is 1


    molecules = data_frame.iloc[:, 0].apply(Chem.MolFromSmiles)

    # Calculate descriptors for each molecule
    descriptors = []
    header_row = ['Molecule']
    for descriptor_name, descriptor_func in Descriptors.descList:
        header_row.append(descriptor_name)

    descriptors.append(header_row)

    for molecule in molecules:
        if molecule is not None:
            descriptor_values = [Chem.MolToSmiles(molecule)]
            for descriptor_name, descriptor_func in Descriptors.descList:
                try:
                    value = descriptor_func(molecule)
                    descriptor_values.append(value)
                except Exception as e:
                    descriptor_values.append(None)
                    print(f"Error calculating {descriptor_name}: {e}")
            descriptors.append(descriptor_values)

    # Create a new DataFrame with descriptors
    descriptors_data_frame = pd.DataFrame(descriptors[1:], columns=descriptors[0])
    data_frame_std = StandardScaler().fit_transform(descriptors_data_frame.iloc[:,1:])
    data_frame_std =  pd.DataFrame(data_frame_std, index=descriptors_data_frame, columns=descriptors_data_frame.columns[1:])

    pca = PCA()
    principalComponents = pca.fit_transform(data_frame_std)
    principalDF = pd.DataFrame(data=principalComponents)

    # Display the combined DataFrame
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    i = 0
    sumVar = 0
    while sumVar<0.9:
        sumVar = cumulative_variance_ratio[i]
        i +=1
    #print(principalDF.iloc[:,:61])
    SMILES = data_frame.iloc[:,0]
    df_smiles = pd.DataFrame(SMILES)
    df_inhib = data_frame.iloc[:,-1]
    df_index = pd.concat([df_smiles,principalDF.iloc[:,:61]], axis=1, join = 'inner')
    PCA_data = pd.concat([df_index,df_inhib], axis=1, join = 'inner')

    PCA_data.to_csv('PCA_data')


filtered_data =open_excel_filter(r"C:\Users\20212807\OneDrive - TU Eindhoven\Advanced programming\Group Assignment\tested_molecules-1.csv")

