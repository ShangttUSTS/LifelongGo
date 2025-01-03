import sys
import os
import csv
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import click as ck
import json
import pandas as pd
from src.utils import Ontology, propagate_annots, get_color
from src.model_use import SharedCoreDeepGATModel,TaskSpecificModel
from src.data import load_ppi_data,load_normal_forms,run_diamond_blastp_and_get_first_result
from src.extract_esm import extract_esm
import torch as th
from gendata.step_5_run_pdb2 import get_pdb_feature
import gzip
import os
import dgl
import requests
from multiprocessing import Pool
from functools import partial
from src.logging import MyLog
myLogging = MyLog().logger
@ck.command()
@ck.option('--in-file', '-if', help='Input FASTA file', default='example/Q5H9Q6.fasta', required=True)
@ck.option('--in-db', '-id', help='Input Diamond DB file', default='initData/swissprot_exp.dmnd', required=True)
@ck.option('--in-pdb', '-ip', help='Input PDB file',  default='example/Q5H9Q6.pdb', required=True)
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--result-root', '-rr', default='results',
    help='Prediction results root')
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
def case_study(in_file, in_db,in_pdb, data_root,result_root,cpd_model, model_dir, threshold, batch_size, device):
    """
    Case study for testing a single data sample.
    """
    fn = os.path.splitext(in_file)[0]
    out_file_esm = f'{fn}_esm.pkl'
    myLogging.info('Process pdb feature')
    pdb2_feature = get_pdb_feature(in_pdb, cpd_model)
    pdb2_feature = list(pdb2_feature)
    myLogging.info('Process ESM2 feature')
    proteins, esm2_data = extract_esm(in_file, out_file=out_file_esm, device=device)
    esm_feature = list(esm2_data)
    combined_feature = np.concatenate((esm_feature[0], pdb2_feature[0]))



    # Define the necessary paths and features based on model_name
    features_length = 2560 + 20
    features_column = 'esm_pdb2'
    ent_models = {
        'mf': 'bp_cc_mf',
        'cc': 'bp_mf_cc',
        'bp': 'cc_mf_bp'
    }
    shared_model = SharedCoreDeepGATModel(shared_input_length=2580, shared_hidden_dim=2560,
                                          shared_embed_dim=2560).to(device)
    output_diamond_file = f"{in_file.split('.')[0]}_result.tsv"
    run_diamond_blastp_and_get_first_result(in_db, in_file, output_diamond_file)
    for ont in ['mf', 'cc', 'bp']:
        myLogging.info(f'Predicting {ont} classes')
        # Load the trained model
        terms_file = f'{data_root}/{ont}/terms.pkl'
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['gos'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        model_file = f'{model_dir}/{ont}_LifeLongGo_esm_pdb2_{ent_models[ont]}_test.th'
        go_norm = f'{data_root}/go.norm'
        n_terms = len(terms_dict)
        _, _, _, _, relations, zero_classes = load_normal_forms(
            go_norm, terms_dict)
        n_rels = len(relations)
        n_zeros = len(zero_classes)

        net = TaskSpecificModel(shared_model, 2580, n_terms, n_zeros, n_rels, device).to(device)


        # Get the the similary graph

        ppi_graph_file = f'ppi_test.bin'
        test_data_file = f'test_data.pkl'

        # Here we assume that test_nids are the indices of the test data nodes.
        train_df = pd.read_pickle(f'data/{ont}/train_data.pkl')
        valid_df = pd.read_pickle(f'data/{ont}/valid_data.pkl')
        test_df = pd.read_pickle(f'data/{ont}/test_data.pkl')

        # Get the proteins and their indices
        proteins = train_df['proteins']
        prot_idx = {v: k for k, v in enumerate(proteins)}
        train_n = len(proteins)

        # Create a mapping for validation and test data as well
        valid_proteins = valid_df['proteins']
        for i, p_id in enumerate(valid_proteins):
            prot_idx[p_id] = train_n + i
        valid_n = len(valid_proteins)

        test_proteins = test_df['proteins']
        for i, p_id in enumerate(test_proteins):
            prot_idx[p_id] = train_n + valid_n + i
        try:
            with open(output_diamond_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row:  # 如果该行不为空，视为符合预期
                        first_result = row
                        myLogging.info(first_result)
                        target_protein = first_result[1]
                        # myLogging.info(target_protein)
                        # Check if the protein_name is in the prot_idx dictionary
                        if target_protein not in prot_idx:
                            myLogging.error(f"Protein {target_protein} not found in the graph.")
                            continue
                        # Get the node id for the given protein_name
                        test_node = prot_idx[target_protein]
                        myLogging.info(f'{ont} Similar Protein:{target_protein}, Node: {test_node}')
                        break
                    else:
                        continue
        except FileNotFoundError:
            myLogging.error(f"File {output_diamond_file} not found.")
        except IOError as e:
            myLogging.error(f"Error reading file: {e}")


        # Loading PPI data
        mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
            data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)
        graph.ndata['feat'][test_node] = th.tensor(combined_feature)
        graph = graph.to(device)
        # myLogging.info(graph)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        test_dataloader = dgl.dataloading.DataLoader(
            graph, [test_node], sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        # Model prediction
        net.load_state_dict(th.load(model_file, map_location=device))
        net.eval()
        with th.no_grad():
            for input_nodes, output_nodes, blocks in test_dataloader:
                logits = net(input_nodes, output_nodes, blocks)
                predicted = logits.detach().cpu().numpy()
                # You can return or myLogging.info predictions here
                # myLogging.info(f"Prediction for test node {test_node}: {predicted}")

        preds = np.array(predicted)
        out_file = f'{result_root}/preds_{ont}_{ent_models[ont]}.txt'
        go_file = f'{data_root}/go.obo'
        go = Ontology(go_file, with_rels=True)
        term_data = {}
        with Pool(2) as p:
            preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
        with open(out_file, 'wt') as f:
            above_threshold = np.argwhere(preds[0] >= threshold).flatten()
            above_threshold = above_threshold[np.argsort(-preds[0][above_threshold])]
            myLogging.info(f'Above threshold:{threshold},Lenth:{len(above_threshold)}')
            for j in above_threshold:
                name = go.get_term(terms[j])['name']
                score = preds[0][j]
                color = get_color(score)
                f.write(f'{fn}\t{ont}\t{terms[j]}\t{name}\t{score:0.3f}\n')
                # 将每个GO术语及其颜色添加到term_data字典中
                term_data[terms[j]] = {'font': color}
        myLogging.info(f'{fn} - {ont} Predicted Saved - {out_file}')

        term_data_json = json.dumps(term_data, indent=4)  # 美化输出
        # myLogging.info(term_data_json)
        # 定义请求的URL
        url = "https://amigo.geneontology.org/visualize"
        # 定义请求的参数
        params = {
            "mode": "amigo",
            "format": "png",
            "term_data_type": "json",
            "inline": "false",
            "term_data": term_data_json}

        # 发送GET请求
        response = requests.get(url, params=params)

        # 检查请求是否成功
        if response.status_code == 200:
            # 保存图片
            with open(f'{result_root}/{ont}_go_visualization.png', 'wb') as f:
                f.write(response.content)
            myLogging.info("Image saved successfully.")
        else:
            myLogging.info("Failed to retrieve image. Status code:", response.status_code)

if __name__ == '__main__':
    case_study()
