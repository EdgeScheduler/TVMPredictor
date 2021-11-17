import os
import json
import ast

from onnx import save_model

class SaveInfo:
    fold_path="Dataset/TVM/models_component/"
    config_name = "dataset.json"
    auto_skip = True

class ModelsRuntimeInfo:
    prefex_fold = "Dataset"
    fold_path="TVM/models_component/"
    config_name = "dataset.json"

def add_model_component(model_name,shape,batch_size,component_dict,dataset_name="dataset.json",fold_path="",auto_skip=True)->dict:
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    dataset_path = os.path.join(fold_path,dataset_name)

    datasets={}
    if os.path.exists(dataset_path):
        with open(dataset_path,'r') as f:
            datasets = json.load(f)
    
    if model_name.lower() not in datasets.keys():
        datasets[model_name.lower()]={}

    if str(shape) not in datasets[model_name.lower()].keys():
        datasets[model_name.lower()][str(shape)]={}

    if str(batch_size) not in datasets[model_name.lower()][str(shape)].keys() or auto_skip is False:
        datasets[model_name.lower()][str(shape)][str(batch_size)]=component_dict

    with open(dataset_path,"w") as f:
        json.dump(datasets,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

    return datasets

def analyze_models_by_runtime_json(deal_function):
    runtime_config = {}
    runtime_config_json_path= os.path.json(ModelsRuntimeInfo.prefex_fold,ModelsRuntimeInfo.fold_path,ModelsRuntimeInfo.config_name)
    if not os.path.exists(runtime_config_json_path):
        print(runtime_config_json_path +" is not found.")
        return False

    with open(runtime_config_json_path,"r") as f:
        runtime_config = json.load(f)

    # 遍历models-runtime数据集的json文件
    for device_name in runtime_config.keys():
        if device_name=="count":
            continue
        
        for object_name in runtime_config[device_name].keys():
            if object_name=="count":
                continue

            for device_id in runtime_config[device_name][object_name].keys():
                if device_id=="count":
                    continue

                for shapes_dimensionality in runtime_config[device_name][object_name][device_id].keys():
                    if shapes_dimensionality=="count":
                        continue
                    
                    for shape in runtime_config[device_name][object_name][device_id][shapes_dimensionality].keys():
                        if shape=="count":
                            continue

                        value = runtime_config[device_name][object_name][device_id][shapes_dimensionality][shape]
                        with open(os.path.join(ModelsRuntimeInfo.prefex_fold, value["file_path"])) as f:
                            line = f.readline()
                            while line is not None and len(line)>0 :
                                batch_size = int(line.split(",")[0])
                                # runtime = float(line.split(",")[1])

                                dshapes = ast.literal_eval(shape.replace("-1",str(batch_size)))
                                flag,model_component_dict = deal_function(object_name,dshapes)
                                if not flag:
                                    print("analyze model component error.")
                                    return False

                                add_model_component(model_name=object_name,batch_size=batch_size,component_dict=model_component_dict,dataset_name=SaveInfo.config_name,fold_path=SaveInfo.fold_path,auto_skip=SaveInfo.auto_skip)

                                line = f.readline()
    return True

# 待实现
def analyze_model_component(model_name,dshape,dtype="float32"):
    return True,{}

def main():
    if analyze_models_by_runtime_json(None):
        print("finish all works.")
    else:
        print("some errors occurr.")

if __name__ == "__main__":
    main()