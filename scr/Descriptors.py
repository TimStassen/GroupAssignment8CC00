
def get_descriptors(filename, outfile):

    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # Read the CSV file using pandas
    data = pd.read_csv(filename)

    # Create lists to store the calculated descriptors
    molecule_smiles = []
    molecule_inhibition = []
    molecule_descriptors = {}

    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        # Extract SMILES and inhibition values from the row
        smiles = row['SMILES']
        inhibition = int(row['ALDH1_inhibition'])

        # Convert the SMILES string to a molecule object using RDKit
        molecule = Chem.MolFromSmiles(smiles)

        # Calculate descriptors for the molecule
        descriptors = {}
        for descriptor_name, descriptor_function in Descriptors.descList:
            descriptor_value = descriptor_function(molecule)
            descriptors[descriptor_name] = descriptor_value

        # Append the SMILES, inhibition, and descriptors to the respective lists
        molecule_smiles.append(smiles)
        molecule_inhibition.append(inhibition)
        molecule_descriptors.update(descriptors)

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({
        'SMILES': molecule_smiles,
        'ALDH1_inhibition': molecule_inhibition,
        **molecule_descriptors  # Unpack the descriptors dictionary using **
    })

    # Save the result DataFrame to a new CSV file
    result_df.to_csv(outfile, index=False)
    return
get_descriptors(filename='tested_molecules-1.csv', outfile='calculated_descriptors.csv')
