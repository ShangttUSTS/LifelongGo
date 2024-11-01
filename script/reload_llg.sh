#!/bin/bash

conda activate llg
jobDir='you Path'
cd $jobDir
script="train_llg.py"
data_dir="data"
batch_size=8
epochs=512
test_data="test"
modes_dir="../models"
result_dir="../tempResults"
sub_ontologies=("mf" "cc" "bp")
combinations=()
for i in "${sub_ontologies[@]}"; do
  for j in "${sub_ontologies[@]}"; do
    for k in "${sub_ontologies[@]}"; do
      if [ "$i" != "$j" ] && [ "$i" != "$k" ] && [ "$j" != "$k" ]; then
        combinations+=("${i}_${j}_${k}")
      fi
    done
  done
done

for combo in "${combinations[@]}"; do
  echo "Running task for ${combo}"
  python $script -dr ${data_dir} -m "LifeLongGo_esm_pdb2" -bs $batch_size -ep $epochs -so $combo -td $test_data -md &modes_dir -rd $result_dir -ld
        rm $result_dir/*.pkl
  wait
done






