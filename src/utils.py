from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
HAS_FUNCTION = 'http://mowl.borg/has_function'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}


params_local = {
    # from params_global_constant
    'device_ids': [1],
    'run_step1_SplitTrainTest_Terms': 'T',
    'run_step2_Train': 'T',
    'run_step3_Test': 'T',
    'run_step4_pkl2fa': 'T',
    'run_step7.1_EvaluateWithoutAlpha': 'T',
    'run_step8.1_PredictWithoutAlpha': 'F',
    'run_step5_Diamond4CrossSpecies': 'F',
    'run_step6_FindAlpha': 'F',
    'run_step7_EvaluateAlpha': 'F',
    'run_step8_PredictAlpha': 'F',
    'aa_ss': 'ss8',
    'train_data': 'ALL00',
    'test_data': 'ALL00',
    'dir0': 'test_TrainALL00_TestALL00_ss8/',
    'path_base': '/scem/work/songfu/py_proj/prot_algo/DeepSS2GO/',
    'path_pub_data': '/scem/work/songfu/py_proj/prot_algo/DeepSS2GO/pub_data/',
    'go_file': 'data/go.obo',
    'GOMinRepeat': 50,
    'TrainTestRatio': 0.95,
    'TrainValidRatio': 0.9,
    'batch_size': 32,
    'MAXLEN': 1024,
    'FC_depth': 0,
    'learning_rate': 0.0003,
    'epochs': 50,
    'load_pretrained_model': 0,
    'load_pretrained_model_addr': 'data/model_checkpoint.pth',
    'EarlyStopping_patience': 6,
    'PROT_LETTER_aa': ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
    'PROT_LETTER_ss8': ['C', 'S', 'T', 'H', 'G', 'I', 'E', 'B'],
    'PROT_LETTER_ss3': ['C', 'E', 'H'],

    # split: params_global_dynamic
    'kernels': [32],
    'filters': [32768],
    'onts': 'all',
}


EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

# CAFA4 Targets
CAFA_TARGETS = set([
    '287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090',
    '10116', '44689', '83333', '99287', '226900', '243273', '284812', '559292'])

def is_cafa_target(org):
    return org in CAFA_TARGETS

def is_exp_code(code):
    return code in EXP_CODES


def get_goplus_defs(filename='data/definitions.txt'):
    plus_defs = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            go_id, definition = line.split(': ')
            go_id = go_id.replace('_', ':')
            definition = definition.replace('_', ':')
            plus_defs[go_id] = set(definition.split(' and '))
    return plus_defs


def propagate_annots(preds, go, terms_dict):
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = preds[j]
        for sup_go in go.get_ancestors(go_id):
            if sup_go in prop_annots:
                prop_annots[sup_go] = max(prop_annots[sup_go], score)
            else:
                prop_annots[sup_go] = score
    for go_id, score in prop_annots.items():
        if go_id in terms_dict:
            preds[terms_dict[go_id]] = score
    return preds


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0
        self.ancestors = {}

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all1.csv types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        if term_id in self.ancestors:
            return self.ancestors[term_id]
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        self.ancestors[term_id] = term_set
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()
        for term_id in terms:
            prop_terms |= self.get_ancestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def read_fasta(filename):
    """
    Reads protein sequences from FASTA file
    Args:
       filename (string or pathlib.Path): FASTA filename
    Returns:
       info (list): List of protein ids
       seqs (list): List of protein sequences
    
    """
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs

def calculate_result(predsEsm, predsPdb, predsInter, alpha, beta):
    for esm, pdb, inter in zip(predsEsm, predsPdb, predsInter):
        yield (1 - alpha - beta) * inter + alpha * esm + beta * pdb


def parse_stream(f, comment=b'#'):
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)

def validate_subontology(value):
    valid_combinations = [
        ['bp', 'cc', 'mf'],
        ['bp', 'mf', 'cc'],
        ['mf', 'bp', 'cc'],
        ['mf', 'cc', 'bp'],
        ['cc', 'mf', 'bp'],
        ['cc', 'bp', 'mf']
    ]
    value_list = value.split('_')
    if value_list in valid_combinations:
        return value_list
    else:
        raise ValueError(f'{value} is not a valid sub-ontology combination')

AALETTER = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
AANUM = len(AALETTER)
AAINDEX = dict()
for i in range(len(AALETTER)):
    AAINDEX[AALETTER[i]] = i + 1
INVALID_ACIDS = set(['U', 'O', 'B', 'Z', 'J', '*'])
MAXLEN = 1000

def is_ok(seq):
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True

def to_tokens(seq):
    tokens = np.zeros((MAXLEN, ), dtype=np.float32)
    l = min(MAXLEN, len(seq))
    for i in range(l):
        tokens[i] = AAINDEX.get(seq[i], 21)
    return tokens

def to_onehot(seq, maxlen=MAXLEN, start=0):
    onehot = np.zeros((22, MAXLEN), dtype=np.float32)
    l = min(MAXLEN, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 21), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot
def get_color(score):
    if score < 0.5:
        return "#F76363"  # 浅红色
    elif score < 0.6:
        return "#F9BE56"  # 浅橙色
    elif score < 0.7:
        return "#F2E659"  # 浅黄色
    elif score < 0.8:
        return "#90D4F2"  # 浅蓝色
    elif score < 0.9:
        return "#56D9CD"  # 浅青色
    else:
        return "#4CC97C"  # 浅绿色

