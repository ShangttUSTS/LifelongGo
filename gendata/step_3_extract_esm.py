import sys
sys.path.append('.')
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

# Extract ESM2 embeddings
from src.extract_esm import extract_esm
from src.logging import MyLog
import pandas as pd
myLogging = MyLog().logger
myLogging.info('Extracting ESM2 embeddings')
prots, esm2_data = extract_esm('../initdata/uniprot_sprot_2024_05.dat.fa', device='cuda:0')
esm2_data = esm2_data.cpu().tolist()
myLogging.info(f"Length of esm2_data: {len(esm2_data)}")
df = pd.read_pickle('../initdata/swissprot_step_1.pkl')
df['esm'] = esm2_data
df.to_pickle('../initdata/swissprot_step3_esm.pkl')
myLogging.info("ESM2 File saved successfully.")