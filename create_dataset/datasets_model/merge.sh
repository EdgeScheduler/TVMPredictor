#!/bin/bash

aim_path="/root/yutian/TVMPredictor/create_dataset/datasets_model/dataset_.json"
jsons_path="/root/yutian/TVMPredictor/create_dataset/datasets_model/*.json"

SCRIPTFOLD="$( cd "$( dirname "$0"  )" && pwd )"
cd $SCRIPTFOLD/../../
echo "run in fold: $PWD"

echo "---start to merge datasets---"

index=1
COUNT=`find $jsons_path -type f | wc -l`
for data in $jsons_path
do
    if [ "$data" == "$aim_path" ];then
        echo "done($index/$COUNT): skip merge $aim_path with itself."
        index=`expr $index + 1`
        continue
    fi

    echo "start to run($index/$COUNT): python3 -u \"$SCRIPTFOLD/../../create_dataset/fix_json/merge_dataset_files.py\" $aim_path $data"
    python3 -u "$SCRIPTFOLD/../../create_dataset/fix_json/merge_dataset_files.py" $aim_path $data
    
    echo -e "done($index/$COUNT): python3 -u \"$data\".\n"
    index=`expr $index + 1`
done 
echo "---finished---"