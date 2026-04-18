#!/usr/bin/env python3
"""
Generate protonated forms from SMILES - standalone utility.
Can process a single SMILES string or a CSV file with a SMILES column.

Usage:
    python generate_protonated.py "CCO"                     # Single molecule
    python generate_protonated.py dataset.csv               # CSV file
    python generate_protonated.py dataset.csv -o output.csv # Custom output
"""
import copy
import argparse
import pandas as pd
import os
from rdkit import Chem
from datetime import datetime

def generate_protonated_forms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []

        protonated_molecules = []

        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)

            if atom.GetSymbol() in ['N', 'O', 'S', 'P'] and atom.GetFormalCharge() == 0:
                try:
                    mol_copy = copy.deepcopy(mol)
                    atom_copy = mol_copy.GetAtomWithIdx(atom_idx)
                    atom_copy.SetFormalCharge(1)
                    Chem.SanitizeMol(mol_copy)
                    prot_smiles = Chem.MolToSmiles(Chem.AddHs(mol_copy), canonical=True)

                    protonated_molecules.append({
                        'protonated_smiles': prot_smiles,
                        'protonation_site_index': atom_idx,
                        'protonation_element': atom.GetSymbol()
                    })

                except:
                    continue

        unique_molecules = []
        seen_smiles = set()
        for mol_data in protonated_molecules:
            if mol_data['protonated_smiles'] not in seen_smiles:
                unique_molecules.append(mol_data)
                seen_smiles.add(mol_data['protonated_smiles'])

        return unique_molecules

    except:
        return []

def process_csv(input_file, smiles_column='smiles'):
    df = pd.read_csv(input_file)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found")

    all_results = []

    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        protonated_forms = generate_protonated_forms(smiles)

        for form in protonated_forms:
            result = {
                'original_index': idx,
                'original_smiles': smiles,
                'protonated_smiles': form['protonated_smiles'],
                'protonation_site_index': form['protonation_site_index'],
                'protonation_element': form['protonation_element']
            }
            all_results.append(result)

    return pd.DataFrame(all_results)

def main():
    parser = argparse.ArgumentParser(description='Generate protonated forms from SMILES')
    parser.add_argument('input', help='SMILES string or CSV file path')
    parser.add_argument('--smiles_column', default='smiles', help='Column name for CSV input')
    parser.add_argument('--output', '-o', help='Output filename')

    args = parser.parse_args()

    # Check if input is a file or SMILES string
    if os.path.exists(args.input):
        # Process CSV file
        print(f"Processing CSV file: {args.input}")
        results_df = process_csv(args.input, args.smiles_column)

        if len(results_df) > 0:
            output_file = args.output or f"protonated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(output_file, index=False)

            print(f"Processed {results_df['original_index'].nunique()} molecules")
            print(f"Generated {len(results_df)} protonated forms")
            print(f"Results saved to: {output_file}")

            element_counts = results_df['protonation_element'].value_counts()
            print("Protonation sites by element:")
            for element, count in element_counts.items():
                print(f"  {element}: {count}")
        else:
            print("No protonated forms generated")

    else:
        # Process single SMILES string
        smiles = args.input
        print(f"Input SMILES: {smiles}")

        protonated_forms = generate_protonated_forms(smiles)

        if protonated_forms:
            print(f"Generated {len(protonated_forms)} unique protonated forms:")

            results = []
            for i, form in enumerate(protonated_forms, 1):
                print(f"{i}. Protonate {form['protonation_element']} at index {form['protonation_site_index']}: {form['protonated_smiles']}")
                results.append({
                    'original_smiles': smiles,
                    'protonated_smiles': form['protonated_smiles'],
                    'protonation_site_index': form['protonation_site_index'],
                    'protonation_element': form['protonation_element']
                })

            if args.output:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
                print(f"Results saved to: {args.output}")
        else:
            print("No protonated forms generated")

if __name__ == "__main__":
    main()
