import click as ck
import csv
import os
import torch as th
import numpy as np
from torch.nn import functional as F
import math
from torch.optim.lr_scheduler import MultiStepLR
from src.torch_utils import EarlyStopping
from src.utils import Ontology, propagate_annots
from src.model_use import SharedCoreDeepGATModel,TaskSpecificModel
from src.data import load_ppi_data,load_normal_forms
from src.metrics import compute_roc, evaluate
from multiprocessing import Pool
from functools import partial
import dgl
from src.utils import validate_subontology
from src.logging import MyLog
myLogging = MyLog().logger
@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--model-dir', '-md', default='models',
    help='Models folder')
@ck.option(
    '--results-dir', '-rd', default='results',
    help='Results folder')
@ck.option(
    '--model-name', '-m', type=ck.Choice([
        'LifeLongGo_esm_pdb2', 'LifeLongGo_esm', 'LifeLongGo_pdb2',
        'LifeLongGo_esmS', 'LifeLongGo_esmS_pdb2']),
    default='LifeLongGo_esm_pdb2',
    help='Prediction model name')
@ck.option(
    '--model-id', '-mi', type=int, required=False)
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'cafa3']),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=8,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=512,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
@ck.option(
    '--sub-ontologies', '-so', default='bp_mf_cc',
    help='Sub-ontologies list (comma-separated)')
def main(data_root, model_dir, results_dir, model_name, model_id, test_data_name, batch_size, epochs, load, device, sub_ontologies):
    """
    This script is used to train LifeLongGo models
    """
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    myLogging.info(f"Is load :{load}")
    ontList = sub_ontologies
    model_name = f'{model_name}_{ontList}'
    myLogging.info(model_name)
    if model_name.find('esmS_pdb2') != -1:
        features_length = 1280 + 20
        features_column = 'esmS_pdb2'
    elif model_name.find('esm_pdb2') != -1:
        features_length = 2560 + 20
        features_column = 'esm_pdb2'
    elif model_name.find('esm') != -1:
        features_length = 2560
        features_column = 'esm'
    elif model_name.find('esmS') != -1:
        features_length = 1280
        features_column = 'esmS'
    elif model_name.find('pdb2') != -1:
        features_length = 20
        features_column = 'pdb2'
    shared_model = SharedCoreDeepGATModel(shared_input_length=features_length, shared_hidden_dim=2560, shared_embed_dim=2560).to(device)
    results = []
    csv_file = f'{results_dir}/{model_name}_predictions_{test_data_name}.csv'
    sub_ontologies = validate_subontology(sub_ontologies)
    myLogging.info(sub_ontologies)
    for ont in sub_ontologies:
        if model_id is not None:
            model_name = f'{model_name}_{model_id}'
        go_file = f'{data_root}/go.obo'
        go_norm_file = f'{data_root}/go.norm'
        model_file = f'{model_dir}/{ont}_{model_name}_{test_data_name}.th'
        out_file = f'{results_dir}/{ont}_{model_name}_predictions_{test_data_name}.pkl'
        go = Ontology(go_file, with_rels=True)
        # Load the datasets
        ppi_graph_file = f'ppi_{test_data_name}.bin'
        test_data_file = f'{test_data_name}_data.pkl'

        mfs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
            data_root, ont, features_length, features_column, test_data_file, ppi_graph_file)
        n_terms = len(terms_dict)

        if features_column == 'prop_annotations':
            features_length = len(mfs_dict)

        valid_labels = labels[valid_nids].numpy()
        test_labels = labels[test_nids].numpy()

        labels = labels.to(device)
        graph = graph.to(device)

        train_nids = train_nids.to(device)
        valid_nids = valid_nids.to(device)
        test_nids = test_nids.to(device)

        _, _, _, _, relations, zero_classes = load_normal_forms(
            go_norm_file, terms_dict)
        n_rels = len(relations)
        myLogging.info(n_rels)
        n_zeros = len(zero_classes)
        myLogging.info(n_zeros)
        net = TaskSpecificModel(shared_model, features_length, n_terms,n_zeros,n_rels, device).to(device)
        myLogging.info(net)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        train_dataloader = dgl.dataloading.DataLoader(
            graph, train_nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        valid_dataloader = dgl.dataloading.DataLoader(
            graph, valid_nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        test_dataloader = dgl.dataloading.DataLoader(
            graph, test_nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)
        optimizer = th.optim.Adam(net.parameters(), lr=1e-5)
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 120], gamma=0.1)
        early_stopping = EarlyStopping(patience=15, verbose=True)
        best_loss = 10000.0
        if shared_model.fisher_information is None:
            shared_model.fisher_information = {}
            for name, param in shared_model.named_parameters():
                shared_model.fisher_information[name] = th.zeros_like(param)
        if not load:
            myLogging.info(f"##########Training  for {ont}...##########")
            for epoch in range(epochs):
                net.train()
                train_loss = 0
                train_steps = int(math.ceil(len(train_nids) / batch_size))
                with ck.progressbar(length=train_steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in train_dataloader:
                        bar.update(1)
                        logits = net(input_nodes, output_nodes, blocks)
                        batch_labels = labels[output_nodes]
                        loss = F.binary_cross_entropy(logits, batch_labels)
                        ewc_loss = shared_model.ewc_loss(shared_model.fisher_information)
                        total_loss = loss + ewc_loss
                        train_loss += loss.detach().item()
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        for name, param in shared_model.named_parameters():
                            shared_model.fisher_information[name] += param.grad ** 2
                    shared_model.save_old_parameters()
                    for name, fisher in shared_model.fisher_information.items():
                        shared_model.fisher_information[name] /= len(train_dataloader)
                train_loss /= train_steps
                myLogging.info('Validation')
                net.eval()
                with th.no_grad():
                    valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                    valid_loss = 0
                    preds = []
                    with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                        for input_nodes, output_nodes, blocks in valid_dataloader:
                            bar.update(1)
                            logits = net(input_nodes, output_nodes, blocks)
                            batch_labels = labels[output_nodes]
                            batch_loss = F.binary_cross_entropy(logits, batch_labels)
                            valid_loss += batch_loss.detach().item()
                            preds = np.append(preds, logits.detach().cpu().numpy())
                    valid_loss /= valid_steps
                    roc_auc = compute_roc(valid_labels, preds)
                    myLogging.info(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    myLogging.info('Saving model')
                    th.save(net.state_dict(), model_file)
                    shared_model.save_old_parameters()
                    myLogging.info('Saving shared_model')
                early_stopping(valid_loss, net)
                if early_stopping.early_stop:
                    myLogging.info("Early stopping")
                    break
                scheduler.step()
        # Loading best model
        myLogging.info('Loading the best model')
        myLogging.info('########## Test the model ##########')
        net.load_state_dict(th.load(model_file))
        num_params = sum(p.numel() for p in net.parameters())
        print(f"Total Params Numbers: {num_params}")
        net.eval()
        with th.no_grad():
            valid_steps = int(math.ceil(len(valid_nids) / batch_size))
            valid_loss = 0
            preds = []
            with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in valid_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    batch_labels = labels[output_nodes]
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
        with th.no_grad():
            test_steps = int(math.ceil(len(test_nids) / batch_size))
            test_loss = 0
            preds = []
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in test_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    batch_labels = labels[output_nodes]
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds.append(logits.detach().cpu().numpy())
                test_loss /= test_steps
            preds = np.concatenate(preds)
            roc_auc = compute_roc(test_labels, preds)
        myLogging.info(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, AUC - {roc_auc}')
        preds = list(preds)
        # Propagate scores using ontology structure
        with Pool(2) as p:
            preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
        test_df['preds'] = preds
        test_df.to_pickle(out_file)
        myLogging.info(f'Test Files Saved - {out_file}')
        myLogging.info('########## evaluate ##########')
        fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match = evaluate(data_root, ont, model_name,
                                                                                         out_file)
        myLogging.info(
            f'model_name:{model_name}, ont:{ont}, batch_size:{batch_size}, epochs:{epochs}, script:{load}, device:{device}')
        myLogging.info(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}, spec: {fmax_spec_match}')
        myLogging.info(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
        myLogging.info(f'AUC: {avg_auc:0.3f}')
        myLogging.info(f'AUPR: {aupr:0.3f}')
        results.append([model_name, ont, fmax, avg_auc, aupr])
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'ont', 'fmax', 'avg_auc', 'aupr'])  # 写入表头
        writer.writerows(results)

if __name__ == '__main__':
    main()
