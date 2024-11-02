#!/bin/bash

conda activate llg
jobDir='you Path'
cd $jobDir
python predict_llg.py -if example/Q5H9Q6.fasta -ip example/Q5H9Q6.pdb