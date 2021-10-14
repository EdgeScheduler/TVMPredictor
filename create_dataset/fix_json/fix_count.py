import json
import os

program_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(program_path,"../datasets/dataset.json") 
datas = {}

with open(json_path,'r') as f:
    datas = json.load(f)

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

with open(json_path,"w") as f:
    json.dump(datas,fp=f,indent=4,separators=(',', ': '),sort_keys=True)