#!/usr/bin/env python
import sys
sys.path.append('.')
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import click as ck
import pandas as pd
import gzip
from tqdm import tqdm

from src.logging import MyLog
myLogging = MyLog().logger

@ck.command()
@ck.option(
    '--string-db-actions-file', '-sdb', default='../initData/protein.actions.v11.0.txt.gz',
    help='String Database Actions file')
@ck.option(
    '--data-file', '-df', default='../initData/swissprot_step5_esm_pdb2.pkl',
    help='Swissprot pandas DataFrame')
@ck.option(
    '--out-file', '-of', default='../initData/swissprot_step6_esm_pdb2_ppi.pkl',
    help='Swissprot pandas DataFrame')
def main(string_db_actions_file, data_file,out_file):
    df = pd.read_pickle(data_file)
    mapping = {}
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Mapping string IDs to proteins")):
        for st_id in row.string_ids:
            mapping[st_id] = row.proteins

    relations = {}
    inters = {}
    with gzip.open(string_db_actions_file, 'rt') as f:
        next(f)
        for line in tqdm(f, desc="Processing STRING DB actions file"):
            it = line.strip().split('\t')
            p1, p2 = it[0], it[1]
            if p1 not in mapping or p2 not in mapping:
                continue
            score = int(it[6])
            if score < 700:
                continue
            p1, p2 = mapping[p1], mapping[p2]
            rel = it[2]
            if rel not in relations:
                relations[rel] = len(relations)
            is_dir = it[4] == 't'
            a_is_act = it[5] == 't'
            if p1 not in inters:
                inters[p1] = set()
            inters[p1].add((rel, p2))
            interactions = []
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Building interactions list")):
            p_id = row.proteins
            myLogging.info(f"Processing {i + 1} of {len(df)} batches: {p_id}")
            if p_id in inters:
                interactions.append(inters[p_id])
            else:
                interactions.append([])
        # df_onlyPPI = pd.DataFrame({'p_id': p_id, 'data': interactions})
        # df_onlyPPI.to_pickle('ppi_feature.pkl')
        df['interactions'] = interactions
        df.to_pickle(out_file)
            
if __name__ == '__main__':
    main()
