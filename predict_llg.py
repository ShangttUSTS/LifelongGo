#!/usr/bin/env python
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import click as ck
import numpy as np
import pandas as pd
import math
import gzip
from src.utils import Ontology
from src.model_use import SharedCoreDeepGATModel,TaskSpecificModel
from src.data import load_normal_forms
from src.extract_esm import extract_esm
import torch as th
from gendata.step_5_run_pdb2 import get_pdb_feature
import os
import dgl
from src.logging import MyLog
myLogging = MyLog().logger
@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', default='example/Q5H9Q6.fasta', required=True)
@ck.option('--in-pdb', '-ip', help='Input PDB file',  default='example/Q5H9Q6.pdb', required=True)
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--cpd-model', '-cm', default='gendata/models',
    help='Prediction model')
@ck.option(
    '--model-dir', '-md', default='models',
    help='Models folder')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=8, help='Batch size for prediction model')
@ck.option(
    '--device', '-d', default='cpu',
    help='Device')
def main(in_file,in_pdb, data_root,cpd_model, model_dir, threshold, batch_size, device):
    fn = os.path.splitext(in_file)[0]
    out_file_esm = f'{fn}_esm.pkl'
    myLogging.info('Process pdb feature')
    pdb2_feature = get_pdb_feature(in_pdb, cpd_model)
    pdb2_feature = list(pdb2_feature)
    myLogging.info('Process ESM2 feature')
    proteins, esm2_data = extract_esm(in_file,out_file=out_file_esm,  device=device)
    esm_feature = list(esm2_data)
    combined_feature = np.concatenate((pdb2_feature[0], esm_feature[0]))
    data = th.zeros((1, 2580), dtype=th.float32)
    data[0, :]= th.FloatTensor(combined_feature)
    # Load GO and read list of all terms
    go_file = f'{data_root}/go.obo'
    go_norm = f'{data_root}/go.norm'
    go = Ontology(go_file, with_rels=True)
    ent_models = ['bp_cc_mf', 'bp_mf_cc', 'mf_bp_cc', 'mf_cc_bp', 'cc_mf_bp', 'cc_bp_mf']
    for ont in ['mf', 'cc', 'bp']:
        myLogging.info(f'Predicting {ont} classes')
        terms_file = f'{data_root}/{ont}/terms.pkl'
        out_file = f'{fn}_preds_{ont}.tsv.gz'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        graph = dgl.graph(([], []))
        graph.add_nodes(1)
        graph = dgl.add_self_loop(graph)
        graph.ndata['feat'] = data
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        predict_dataloader = dgl.dataloading.DataLoader(
            graph,
            th.arange(graph.number_of_nodes()),  # 索引所有节点,
            sampler,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        n_terms = len(terms_dict)
        _, _, _, _, relations, zero_classes = load_normal_forms(
            go_norm, terms_dict)
        n_rels = len(relations)
        n_zeros = len(zero_classes)
        sum_preds = np.zeros((len(proteins), n_terms), dtype=np.float32)
        shared_model = SharedCoreDeepGATModel(shared_input_length=2580, shared_hidden_dim=2560,
                                              shared_embed_dim=2560).to(device)
        model = TaskSpecificModel(shared_model, 2580, n_terms, n_zeros, n_rels, device).to(device)
        for mn in ent_models:
            model_file = f'{model_dir}/{ont}_LifeLongGo_esm_pdb2_{mn}.th'
            myLogging.info(model_file)
            model.load_state_dict(th.load(model_file, map_location=device))
            model.eval()
            with th.no_grad():
                steps = int(math.ceil(len(proteins) / batch_size))
                preds = []
                with ck.progressbar(length=steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in predict_dataloader:
                        bar.update(1)
                        logits = model(input_nodes, output_nodes, blocks)
                        preds.append(logits.detach().cpu().numpy())
                preds = np.concatenate(preds)
            sum_preds += preds
        preds = sum_preds / len(ent_models)
        with gzip.open(out_file, 'wt') as f:
            for i in range(len(proteins)):
                above_threshold = np.argwhere(preds[i] >= threshold).flatten()
                for j in above_threshold:
                    name = go.get_term(terms[j])['name']
                    f.write(f'{proteins[i]}\t{terms[j]}\t{name}\t{preds[i, j]:0.3f}\n')


if __name__ == '__main__':
    main()