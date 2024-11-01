import sys

sys.path.append('.')
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from src.logging import MyLog
myLogging = MyLog().logger

import click as ck
import torch
import torch.nn as nn
from src.gvp.data import BatchSampler, Dataset, ProteinGraphDataset
from src.gvp.models import CPDModel
from datetime import datetime
import tqdm, os
from torch_geometric.loader import DataLoader
import pickle
from gendata.step_4_pdbPraseToJson import parse_pdb
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pdb_feature(in_pdb,cpd_model_dir):
    data_list = []
    pdb_json = 'swissprot_pdb.json'
    with open(pdb_json, 'w') as json_file:
        data = parse_pdb(in_pdb)
        data_list.append(data)
        json.dump(data, json_file, separators=(',', ':'))
        json_file.write('\n')  # Add newline to separate JSON objects
    node_dim = (100, 16)
    edge_dim = (32, 1)
    path = f"{cpd_model_dir}/CPDModel.pt"
    model = CPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    model.load_state_dict(torch.load(path,map_location='cpu'))
    dataloader = lambda x: DataLoader(x, num_workers=4,
                                      batch_sampler=BatchSampler(x.node_counts, max_nodes=1000))
    # myLogging.info("Loading dataset")
    data = Dataset(path=pdb_json)
    data_list_1 = []
    dataset = ProteinGraphDataset(data.data)
    # myLogging.info("generate pdb2 feature")
    train_loader = dataloader(dataset)
    t = tqdm.tqdm(train_loader.dataset)
    for batch in t:
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        logits = model(h_V, batch.edge_index, h_E, seq=batch.seq)
        logits, seq = logits[batch.mask], batch.seq[batch.mask]
        average_feature = torch.mean(logits, dim=0).tolist()
        data_list_1.append(average_feature)
        torch.cuda.empty_cache()
    return data_list_1


@ck.command()
@ck.option(
    '--models-dir', '-md', default='./models/',
    help='directory to save trained models, default=./models/')
@ck.option(
    '--in-file', '-if', default='../initdata/swissprot_step3_esm.pkl',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--pdb-json', '-pj', default='swissprot_pdb.json',
    help='UniProt/SwissProt knowledgebase file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='../initData/swissprot_step5_esm_pdb2.pkl',
    help='Result file with a list of proteins, sequences and pdb2')
@ck.option(
    '--num-workers', '-nm', type=int, default=4,
    help='number of threads for loading data, default=4')
@ck.option(
    '--max-nodes', '-mn', type=int, default=3000,
    help='max number of nodes per batch, default=3000')
def main(in_file, pdb_json, out_file, models_dir, num_workers, max_nodes):
    node_dim = (100, 16)
    edge_dim = (32, 1)
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    model_id = int(datetime.timestamp(datetime.now()))
    dataloader = lambda x: DataLoader(x, num_workers=num_workers,
                                      batch_sampler=BatchSampler(x.node_counts, max_nodes=max_nodes))
    model = CPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    myLogging.info("Loading dataset")
    data = Dataset(path=pdb_json)
    dataset = ProteinGraphDataset(data.data)
    myLogging.info("generate pdb2 feature")
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    train_loader = dataloader(dataset)
    optimizer = torch.optim.Adam(model.parameters())
    data2 = loop(model, train_loader, optimizer=optimizer)
    path = f"{models_dir}/{model_id}.pt"
    torch.save(model.state_dict(), path)
    data['pdb2'] = data2
    # pdb2_feature = pd.DataFrame(data2)
    # pdb2_feature.to_pickle('pdb2_feature.pkl')
    myLogging.info(f"PDB2 Shape: ({len(data2)}, {len(data2[1])})")
    data.to_pickle(out_file)
    myLogging.info(f'File saved to {out_file}')

def loop(model, dataloader, optimizer=None):
    t = tqdm.tqdm(dataloader.dataset)
    loss_fn = nn.CrossEntropyLoss()
    data = []
    for batch in t:
        if optimizer: optimizer.zero_grad()
        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        logits = model(h_V, batch.edge_index, h_E, seq=batch.seq)
        logits, seq = logits[batch.mask], batch.seq[batch.mask]
        loss_value = loss_fn(logits, seq)
        if optimizer:
            loss_value.backward()
            optimizer.step()
        average_feature = torch.mean(logits, dim=0).tolist()
        data.append(average_feature)
        torch.cuda.empty_cache()
    return data
if __name__ == "__main__":
    main()
