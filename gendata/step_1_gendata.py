#!/usr/bin/env python
import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import click as ck
import pandas as pd
import gzip
from src.utils import Ontology, is_exp_code, is_cafa_target
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from src.logging import MyLog
myLogging = MyLog().logger
@ck.command()

@ck.option(
    '--swissprot-file', '-sf', default='../initdata/uniprot_sprot_2024_05.dat.gz',
      help='Uniport Training  file in text format (archived)')
@ck.option(
    '--out-file', '-o', default='../initdata/swissprot_step_1.pkl',
    help='Result file with a list of proteins, sequences and annotations')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device for ESM2 model')
def main(swissprot_file, out_file, device):
    myLogging.info(f'Parsing UniProt data and creating the DataFrame: {swissprot_file}')
    go = Ontology('../initData/go.obo', with_rels=True)
    proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros, AlphaFoldDB_id = load_data(swissprot_file)
    myLogging.info(f'ProteinNum {len(proteins)}')
    df = pd.DataFrame({
        'proteins': proteins,
        'accessions': accessions,
        'genes': genes,
        'sequences': sequences,
        'annotations': annotations,
        'string_ids': string_ids,
        'orgs': orgs,
        'interpros': interpros,
        'AlphaFoldDB_id': AlphaFoldDB_id
    })

    myLogging.info('Filtering proteins with experimental annotations')
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        # Ignore proteins without experimental annotations
        if len(annots) == 0:
            continue
        index.append(i)
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index()
    df['exp_annotations'] = annotations

    prop_annotations = []
    for i, row in df.iterrows():
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_ancestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations

    cafa_target = []
    for i, row in enumerate(df.itertuples()):
        if is_cafa_target(row.orgs):
            cafa_target.append(True)
        else:
            cafa_target.append(False)
    df['cafa_target'] = cafa_target

    # Save sequences to a FASTA file
    myLogging.info('Save sequences to a FASTA file')
    fasta_file = os.path.splitext(swissprot_file)[0] + '.fa'
    if fasta_file is not None and os.path.exists(fasta_file):
        myLogging.info('FASTA File HAVE EXISTED')
    else:
        with open(fasta_file, 'w') as f:
            for row in df.itertuples():
                record = SeqRecord(
                    Seq(row.sequences),
                    id=row.proteins,
                    description=''
                )
                SeqIO.write(record, f, 'fasta')
        myLogging.info('FASTA File HAVE SAVED SUCCESSFULLY')

    df.to_pickle(out_file)
    myLogging.info('Successfully saved %d proteins' % (len(df),) )


def load_data(swissprot_file):
    """
    Parses UniProtKB data file and loads list of proteins and their
    annotations to lists
    Args:
       swissprot_file (string): A path to the data file
    Returns:
       Tuple of 9 lists (proteins, accessions, sequences, string_ids, orgs, genes, interpros, AlphaFoldDBId )
    """

    proteins = list()
    accessions = list()
    sequences = list()
    annotations = list()
    string_ids = list()
    orgs = list()
    genes = list()
    interpros = list()
    AlphaFoldDB_id = list()
    with gzip.open(swissprot_file, 'rt') as f:
        prot_id = ''
        prot_ac = ''
        seq = ''
        org = ''
        annots = list()
        strs = list()
        iprs = list()
        gene_id = ''
        AFDBId = ''
        for line in f:
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if AFDBId != '' and prot_id != '':
                    proteins.append(prot_id)
                    accessions.append(prot_ac)
                    sequences.append(seq)
                    annotations.append(annots)
                    string_ids.append(strs)
                    orgs.append(org)
                    genes.append(gene_id)
                    interpros.append(iprs)
                    AlphaFoldDB_id.append(AFDBId)
                prot_id = items[1]
                annots = list()
                strs = list()
                iprs = list()
                seq = ''
                gene_id = ''
                AFDBId = ''
            elif items[0] == 'AC' and len(items) > 1:
                prot_ac = items[1]
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
                elif items[0] == 'STRING':
                    str_id = items[1]
                    strs.append(str_id)
                elif items[0] == 'AlphaFoldDB':
                    AFDBId = items[1]
                elif items[0] == 'GeneID':
                    gene_id = items[1]
                elif items[0] == 'InterPro':
                    ipr_id = items[1]
                    iprs.append(ipr_id)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq

        proteins.append(prot_id)
        accessions.append(prot_ac)
        sequences.append(seq)
        annotations.append(annots)
        string_ids.append(strs)
        orgs.append(org)
        genes.append(gene_id)
        interpros.append(iprs)
        AlphaFoldDB_id.append(AFDBId)
    return proteins, accessions, sequences, annotations, string_ids, orgs, genes, interpros, AlphaFoldDB_id
if __name__ == '__main__':
    main()
