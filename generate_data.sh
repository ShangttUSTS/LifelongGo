#!/bin/bash
data_root=initData
gendata_root = gendata
cd $gendata_root
python step_1_gendata.py
python step_2_downLoadPDB.py
python step_3_extract_esm.py
python step_4_pdbPraseToJson.py
python step_5_run_pdb2.py
python step_6_ppi_data.py
echo "Creating Diamond database and compute similarities"
diamond makedb --in $data_root/swissprot_exp_2024_05.fa --db $data_root/swissprot_exp.dmnd
diamond blastp --very-sensitive -d $data_root/swissprot_exp.dmnd -q $data_root/uniprot_sprot_2024_05.dat.fa  --outfmt 6 qseqid sseqid bitscore pident > $data_root/swissprot_exp.sim
python step_7_splitData.py
python step_8_ppi_save_graph.py
groovy step_9_Normalizer.groovy -i data/go.owl -o data/go.norm
echo "Data genrate successfully"
