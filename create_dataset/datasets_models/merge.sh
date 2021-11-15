#!/bin/bash

aim_path="/home/yutian/TVMPredictor/create_dataset/datasets_models/dataset_auto.json"
jsons_path="/home/yutian/TVMPredictor/create_dataset/datasets_models/*.json"

SCRIPTFOLD="$( cd "$( dirname "$0"  )" && pwd )"
cd $SCRIPTFOLD/../../
echo "run in fold: $PWD"

echo "---start to merge datasets---"

index=1
COUNT=`find $jsons_path -type f | wc -l`
for data in $jsons_path
do
    if [ "$data" == "$aim_path" ];then
        echo -e "done($index/$COUNT): skip merge $aim_path with itself.\n"
        index=`expr $index + 1`
        continue
    fi

    echo "start to run($index/$COUNT): python3 -u $SCRIPTFOLD/../../create_dataset/fix_json/merge_dataset_files.py $aim_path $data"
    python3 -u $SCRIPTFOLD/../../create_dataset/fix_json/merge_dataset_files.py $aim_path $data
    
    echo -e "done($index/$COUNT).\n"
    index=`expr $index + 1`
done 
echo "---finished---"