import json
import os
import copy

def fix_dict_count(datas):
    count_total = 0
    for device_name in datas.keys():
        if device_name=="count":
            continue
        
        count_device_name = 0
        for op_name in datas[device_name].keys():
            if op_name=="count":
                continue
            
            count_op_name = 0
            for device_id in datas[device_name][op_name].keys():
                if device_id=="count":
                    continue
                
                count_device_id = 0
                for shapes_dimensionality in datas[device_name][op_name][device_id].keys():
                    if shapes_dimensionality=="count":
                        continue

                    datas[device_name][op_name][device_id][shapes_dimensionality]["count"] = len(datas[device_name][op_name][device_id][shapes_dimensionality].keys()) - 1
                    count_device_id += datas[device_name][op_name][device_id][shapes_dimensionality]["count"]

                datas[device_name][op_name][device_id]["count"] = count_device_id
                count_op_name += datas[device_name][op_name][device_id]["count"]
            
            datas[device_name][op_name]["count"] = count_op_name
            count_device_name += datas[device_name][op_name]["count"]   
        
        datas[device_name]["count"] = count_device_name
        count_total += datas[device_name]["count"]

    datas["count"] = count_total
    return datas

def delete_unexist_config(log_file="Datasets/TVM/datasets/dataset.json",prifex_fold="") ->None:
    '''
    delete un-exist dataset-items from json config
    '''

    # 去除json中有记录，但是实际上被删除的条目
    log_dict = {}
    if os.path.exists(log_file):
        with open(log_file,'r') as f:
            log_dict = json.load(f)

    result = copy.deepcopy(log_dict)

    for device_name,device_dict in log_dict.items():
        if device_name=="count":
            continue
        for function_name,function_dict in device_dict.items():
            if function_name=="count":
                continue
            for device_type,device_type_dict in function_dict.items():
                if device_type=="count":
                    continue
                for shape_dimensionality,shape_dimensionality_dict in device_type_dict.items():
                    if shape_dimensionality=="count":
                        continue
                    for shapes_str,value in shape_dimensionality_dict.items():
                        if shapes_str=="count":
                            continue
                        if not os.path.exists(os.path.join(prifex_fold, value["file_path"])):
                            del result[device_name][function_name][device_type][shape_dimensionality][shapes_str]           # 删除对应条目

                            result[device_name][function_name][device_type][shape_dimensionality]["count"]-=1
                            # 没有条目时，删除维度信息
                            if result[device_name][function_name][device_type][shape_dimensionality]["count"]<=0:
                                del result[device_name][function_name][device_type][shape_dimensionality] 
       
                            result[device_name][function_name][device_type]["count"]-=1
                            # 没有条目时，删除硬件信息
                            if result[device_name][function_name][device_type]["count"]<=0:
                                del result[device_name][function_name][device_type]

                            result[device_name][function_name]["count"]-=1
                            # 没有条目时，删除算子信息
                            if result[device_name][function_name]["count"]<=0:
                                del result[device_name][function_name]

                            result[device_name]["count"]-=1
                            # 没有条目时，删除设备信息
                            if result[device_name]["count"]<=0:
                                del result[device_name]

                            result["count"] -= 1 
                            if result["count"] <= 0:
                                result={"count": 0}
               
    with open(log_file,"w") as f:
        json.dump(result,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

# 将字典2合并到字典1
def merge_dict(dict_1,dict_2):
    if "file_path" in dict_2.keys():
        return

    for key in dict_2.keys():
        if key=="count":
            continue
        elif key not in dict_1.keys():
            dict_1[key] = dict_2[key]
            if "file_path" in dict_2[key].keys():
                dict_1["count"]+=1
            else:
                dict_1["count"]+=dict_2[key]["count"]
        else:
            merge_dict(dict_1[key],dict_2[key])
        
def merge(source_file1,source_file2,aim_file="dataset.json"):
    source_data1={}
    if not os.path.exists(source_file1):
        source_data1 = {"count":0}
    else:
        with open(source_file1,'r') as f:
            source_data1 = json.load(f)
    
    if not (os.path.exists(source_file2) and source_file2.endswith(".json")):
        print("file: <" + source_file2 + "> is not one exist json file.")
        return

    source_data2={}
    with open(source_file2,'r') as f:
        source_data2 = json.load(f)

    merge_dict(source_data1,source_data2)
    # 再次校准count值
    source_data1 = fix_dict_count(source_data1)
    with open(aim_file,"w") as f:
        json.dump(source_data1,fp=f,indent=4,separators=(',', ': '),sort_keys=True)