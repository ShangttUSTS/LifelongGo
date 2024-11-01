#!/usr/bin/env python
import json
import sys,os
sys.path.append('.')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from Bio import PDB
import pickle
from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
from src.logging import MyLog
myLogging = MyLog().logger
def parse_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file)

    seq = ""
    coords = {
        "N": [],
        "CA": [],
        "C": [],
        "O": []
    }
    num_chains = len(structure[0].child_list)
    name = os.path.basename(pdb_file).split('.')[0]

    def format_coord(coord):
        return [round(x, 3) if not isinstance(x, float) or not (x != x) else float('nan') for x in coord]
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    seq += protein_letters_3to1.get(residue.resname, 'X')
                    atom_N = residue['N']
                    atom_CA = residue['CA']
                    atom_C = residue['C']
                    atom_O = residue['O']
                    coords["N"].append(format_coord(atom_N.coord.tolist()) if atom_N else [float('nan')] * 3)
                    coords["CA"].append(format_coord(atom_CA.coord.tolist()) if atom_CA else [float('nan')] * 3)
                    coords["C"].append(format_coord(atom_C.coord.tolist()) if atom_C else [float('nan')] * 3)
                    coords["O"].append(format_coord(atom_O.coord.tolist()) if atom_O else [float('nan')] * 3)
    return {
        "seq": seq,
        "coords": coords,
        "num_chains": num_chains,
        "name": name
    }

def pdb_to_json(AlphaFoldDB_id, pdb_folder, output_file):
    data_list = []
    with open(output_file, 'w') as json_file:
        for index, pdb_file in enumerate(AlphaFoldDB_id, start=1):
            myLogging.info(f"正在处理第 {index} 个文件: {pdb_file}")
            data = parse_pdb(f'{pdb_folder}/{pdb_file}.pdb')
            data_list.append(data)
            json.dump(data, json_file, separators=(',', ':'))
            json_file.write('\n')  # Add newline to separate JSON objects

def main():
    with open('../initData/swissprot_exp_2024_05.pkl', 'rb') as f:
        df = pickle.load(f)
    AlphaFoldDB_id = df['AlphaFoldDB_id']
    pdb_folder = '../data/PDB'
    # pdb_files = ['file1.pdb', 'file2.pdb']  # Replace with your PDB file paths
    output_file = 'swissprot_pdb.json'  # Replace with your output directory
    pdb_to_json(AlphaFoldDB_id, pdb_folder, output_file)

if __name__ == '__main__':
    main()

