import sys
sys.path.append('.')
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.logging import MyLog
myLogging = MyLog().logger
import pickle
import urllib3
urllib3.disable_warnings()

with open('../initdata/swissprot_step_1.pkl', 'rb') as f:
    data = pickle.load(f)
    AlphaFoldDB_id = data['AlphaFoldDB_id']
not_exist_list = []
downloaded_count = 0
headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'}
PDB_folder = '../data/PDB'
max_retries = 5
retry_delay = 2  # seconds
max_workers = 20  # 并发请求数量

# 创建PDB文件夹（如果不存在）
os.makedirs(PDB_folder, exist_ok=True)


def download_pdb_file(prot_id):
    file_path = os.path.join(PDB_folder, f'{prot_id}.pdb')

    if os.path.exists(file_path):
        myLogging.info(f'{file_path} already exists. Skipping download.')
        return None

    url = f'https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb'
    myLogging.info(f'Start downloading {prot_id} from {url}')

    success = False
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, verify=False, proxies={"http": None, "https": None})

            if r.status_code == 200:
                if r.text and r.text[1] == '?':
                    myLogging.info(f'{prot_id} has no PDB File')
                    return prot_id
                else:

                    with open(file_path, 'w') as file:
                        for line in r.text.splitlines():
                            file.write(line + '\n')
                    success = True
                    global downloaded_count
                    downloaded_count += 1
                    break
            else:
                myLogging.warning(f'Failed to download {prot_id}. HTTP status code: {r.status_code}')
                return prot_id

        except (ConnectionResetError, requests.exceptions.RequestException) as e:
            myLogging.warning(f'Attempt {attempt + 1} failed for {prot_id} with error: {e}')
            time.sleep(retry_delay)

    if not success:
        myLogging.error(f'Failed to download {prot_id} after {max_retries} attempts')
        return prot_id

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_prot_id = {executor.submit(download_pdb_file, prot_id): prot_id for prot_id in AlphaFoldDB_id}

    for future in as_completed(future_to_prot_id):
        prot_id = future_to_prot_id[future]
        try:
            result = future.result()
            if result:
                not_exist_list.append(result)
        except Exception as exc:
            myLogging.error(f'{prot_id} generated an exception: {exc}')

myLogging.info(not_exist_list)
myLogging.info(f'Unsuccessfully downloaded {len(not_exist_list)} proteins')
myLogging.info(f'Successfully downloaded {downloaded_count} PDB files')