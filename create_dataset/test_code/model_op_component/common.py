import os
import json

def add_model_component(model_name,dshapes,component_dict,dataset_name="dataset.json",fold_path="",auto_skip=True)->dict:
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    dataset_path = os.path.join(fold_path,dataset_name)

    datasets={}
    if os.path.exists(dataset_path):
        with open(dataset_path,'r') as f:
            datasets = json.load(f)
    
    if model_name.lower() not in datasets.keys():
        datasets[model_name.lower()]={}

    if str(dshapes) not in datasets[model_name.lower()].keys() or auto_skip is False:
        datasets[model_name.lower()][str(dshapes)]=component_dict

    with open(dataset_path,"w") as f:
        json.dump(datasets,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

    return datasets