#!/bin/bash

# 注意：需要修改 line: 8~11 的值才能使用，此脚本仅为模板！！！

SCRIPTFOLD="$( cd "$( dirname "$0"  )" && pwd )"

# 指定参数, 执行路径应该包含create_dataset文件夹
cd $SCRIPTFOLD/../../
aim_path="/home/yutian/TVMPredictor/Datasets/TVM/datasets_models/dataset_auto.json"
jsons_path="/home/yutian/TVMPredictor/Datasets/TVM/datasets_models/*.json"
fix_json_scripy_path="/home/yutian/TVMPredictor/create_dataset/fix_json/merge_dataset_files.py"

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

    echo "start to run($index/$COUNT): python3 -u $fix_json_scripy_path $aim_path $data"
    python3 -u $fix_json_scripy_path $aim_path $data
    
    echo -e "done($index/$COUNT).\n"
    index=`expr $index + 1`
done 
echo "---finished---"