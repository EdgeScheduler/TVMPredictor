import json
import os
from create_dataset.fix_json.common import fix_dict_count

program_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(program_path,"../../Datasets/TVM/datasets/dataset.json") 
datas = {}
with open(json_path,'r') as f:
    datas = json.load(f)

with open(json_path,"w") as f:
    json.dump(fix_dict_count(datas),fp=f,indent=4,separators=(',', ': '),sort_keys=True)