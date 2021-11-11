import json
import os

# 将字典2合并到字典1
def merge_dict(dict_1,dict_2):
    if "file_path" in dict_2.keys():
        return

    for key in dict_2.keys():
        if key not in dict_1.keys():
            dict_1[key] = dict_2[key]
            dict_1["count"]+=dict_2[key]["count"]
        elif key=="count":
            continue
        else:
            merge_dict(dict_1[key],dict_2[key])
        
def merge(source_file1,source_file2,aim_file="dataset.json"):
    if not (os.path.exists(source_file1) and source_file1.endswith(".json")):
        print("file: <" + source_file1 + "> is not one exist json file.")
    
    if not (os.path.exists(source_file2) and source_file2.endswith(".json")):
        print("file: <" + source_file2 + "> is not one exist json file.")
    
    source_data1={}
    with open(source_file1,'r') as f:
        source_data1 = json.load(f)

    source_data2={}
    with open(source_file2,'r') as f:
        source_data2 = json.load(f)

    merge_dict(source_data1,source_data2)
    with open(aim_file,"w") as f:
        json.dump(source_data1,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

merge("/home/yutian/TVMPredictor/create_dataset/datasets/dataset.json","/home/yutian/TVMPredictor/create_dataset/datasets_model/dataset.json","/home/yutian/TVMPredictor/create_dataset/fix_json/dataset.json")