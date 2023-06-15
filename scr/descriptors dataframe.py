import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
def open_excel_filter(filepath):
    # Specify the path to the CSV file
    csv_file_path = r"C:\Users\20212807\OneDrive - TU Eindhoven\Advanced programming\Group Assignment\tested_molecules-1.csv"
    # Read the CSV file
    data_frame = pd.read_csv(csv_file_path)
    # Filter the rows where the right column value is 1
    filtered_data_frame = data_frame[data_frame.iloc[:, 1] == 1]

    molecules = filtered_data_frame.iloc[:, 0].apply(Chem.MolFromSmiles)

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

    # Combine the original DataFrame with the descriptors DataFrame
    combined_data_frame = pd.concat([data_frame, descriptors_data_frame], axis=1)

    # Display the combined DataFrame
    print(combined_data_frame.iloc[:])


filtered_data =open_excel_filter(r"C:\Users\20212807\OneDrive - TU Eindhoven\Advanced programming\Group Assignment\tested_molecules-1.csv")




