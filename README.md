# LIFELONGGO: Protein function predicted based lifelong-learning with migration and tertiary Structure information
`Intro`

This repository contains script which were used to build and train the
LIFELONGGO model together with the scripts for evaluating the model's
performance.c

# Dependencies
* The code was developed and tested using python 3.7.
* `pip xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`
* Clone the repository: `git clone https://github.com/ShangttUSTS/LifelongGo.git`
* Create virtual environment with Conda or python3-venv module. 
* `conda create -n llg python=3.7.16`
* 1. Install requirements:
  ```
  pip install -r requirements.txt
  ```
* 2. Install PyTorch:
  ```
  pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
  ```
* 3. Install DGLGPU: ``
   ```
  pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html 
  conda install -c dglteam/label/cu116 dgl
  ```
* 4. Install MDTRAJ:\
  ```
  pip install mdtraj==1.9.9
   ```
* 5. Install torch-geometric:\
  ```
  pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
  pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
  pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
  pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cu116.html
  pip install torch-geometric
  ```

# Running LIFELONGGO model
Follow these instructions to obtain predictions for your proteins. You'll need
around 30Gb storage and a GPU with >16Gb memory (or you can use CPU)

# Generating the data
The data used in to train our models are available for download. However,
if you like to generate a new dataset follow these steps:
* Download [Gene Ontology](https://geneontology.org/docs/download-ontology/).
* Download [UniProt-KB](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz)
* Download [StringDB v11.0](https://stringdb-static.org/download/protein.actions.v11.0.txt.gz)
```
mkdir initdata
cd initdata
wget https://purl.obolibrary.org/obo/go.obo
wget https://purl.obolibrary.org/obo/go.owl
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
mv uniprot_sprot.dat.gz uniprot_sprot_2024_05.dat.gz
wget https://stringdb-static.org/download/protein.actions.v11.0.txt.gz
```
* Install [Groovy](https://groovy-lang.org/install.html)
* Install [Diamond](https://github.com/bbuchfink/diamond/wiki/2.-Installation)
* Run data generation script: \
  `sh generate_data.sh`

# Training the models
* Examples:
  - Train a single LifelongGO with _MF,BP,CC_ order prediction model which uses _ESM_ & _PDB2_ features \
    `python train_llg.py -m LifeLongGo_esm_pdb2 -bs 8 -ep 512 -so mf,bp,cc`
  - Train a single LifelongGO with _MF,BP,CC_ order prediction model which uses _ESM2_ embeddings \
    `python train_llg.py -m LifeLongGo_esm -bs 8 -ep 512 -so mf,bp,cc`
  - Train a single LifelongGO with _BP,CC,MF_ order prediction model which uses _ESM_ & _PDB2_ embeddings \
    `python train_llg.py -m LifeLongGo_esm -bs 8 -ep 512 -so bp,cc,mf`
  - Train LifelongGO with All order prediction model which uses _ESM_ & _PDB2_ embeddings \
    `sh script/train_llg.sh`

# Citation

If you use LIFELONGGO for your research, or incorporate our learning
algorithms in your work, please cite: 
