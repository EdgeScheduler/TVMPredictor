import json
import os
import sys
import datetime

def op_filter(key,value):
    if value["avg"]>0.05:
        return True

def select_op(my_dict:dict,function):
    return {k: v for k, v in my_dict.items() if function(k,v)}

base_dir=os.path.dirname(__file__)
sys.path.append(base_dir)
print(base_dir)

csv_datas_dict = os.path.join(base_dir,"datas_git/TVMProfiler/model_src/data/")
csv_file_names = os.listdir(csv_datas_dict)

# 从csdv读取数据
datas = {}
for csv_file_name in csv_file_names:
    with open(os.path.join(csv_datas_dict, csv_file_name), "r") as f:
        for line in f:
            line = line.replace("\n", "")                # 每行去掉换行符
            data = line.split(",")                       # 按分隔符分割
            if data[2][-1] != "%":
                continue

            if str(data[0]) not in datas.keys():
                datas[str(data[0])]={}
                datas[str(data[0])]["min"] = float(data[2].strip('%'))/100.0
                datas[str(data[0])]["max"] = float(data[2].strip('%'))/100.0
                datas[str(data[0])]["avg"] = float(data[2].strip('%'))/100.0
                datas[str(data[0])]["count"] = 1
            else:
                datas[str(data[0])]["min"] = min(datas[str(data[0])]["min"], float(data[2].strip('%'))/100.0)
                datas[str(data[0])]["max"] = max(datas[str(data[0])]["max"], float(data[2].strip('%'))/100.0)
                datas[str(data[0])]["count"] = int(datas[str(data[0])]["count"]) + 1
                datas[str(data[0])]["avg"] = (datas[str(data[0])]["avg"]*(int(datas[str(data[0])]["count"])-1) + float(data[2].strip('%'))/100.0)/int(datas[str(data[0])]["count"])

save_json={}
save_json["total"]={}
save_json["total"]["count"] = len(datas.keys())
save_json["total"]["list"] = list(datas.keys())

save_json["selected"]={}
save_json["selected"]["count"] = len(select_op(datas,op_filter).keys())
save_json["selected"]["list"] = list(select_op(datas,op_filter).keys())

save_json["origin_data"]=datas

with open(os.path.join(base_dir,"op_statistics.json"),"w") as f:
        json.dump(save_json,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

print("--totoal:")
print("count:",len(datas.keys()))
print("list:",list(datas.keys()))

print("--selected:")
print("count:",len(select_op(datas,op_filter).keys()))
print("list:",list(select_op(datas,op_filter).keys()))
