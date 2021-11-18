import os
import json
import ast
from TVMProfiler.model_src.analyze_componnet import get_op_info
import tvm
import traceback
import threading
class SaveInfo:
    fold_path="Datasets/TVM/models_component/"
    config_name = "dataset.json"
    auto_skip = True

class ModelsRuntimeInfo:
    prefix_fold = "Datasets/"
    fold_path="TVM/datasets_models/"
    config_name = "dataset_auto.json"

TVM_INIT=False

def add_model_component(model_name,shape,batch_size,component_dict,dataset_name="dataset.json",prefix_dataset_name="",fold_path="",auto_skip=True)->dict:
    dataset_name = prefix_dataset_name + dataset_name

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

def exist_in_dataset(model_name,shape,batch_size,dataset_name="dataset.json",prefix_dataset_name="",fold_path="")->bool:
    dataset_name = prefix_dataset_name+dataset_name
    if not os.path.exists(fold_path):
        return False
    dataset_path = os.path.join(fold_path,dataset_name)

    datasets={}
    if os.path.exists(dataset_path):
        with open(dataset_path,'r') as f:
            datasets = json.load(f)
    else:
        return False

    if model_name.lower() not in datasets.keys():
        return False

    if str(shape) not in datasets[model_name.lower()].keys():
        return False

    if str(batch_size) not in datasets[model_name.lower()][str(shape)].keys():
        return False

    return True

def analyze_models_by_runtime_json(deal_function,object_names=[],prefix_dataset_name="",print_info=True):
    global TVM_INIT

    runtime_config = {}
    runtime_config_json_path= os.path.join(ModelsRuntimeInfo.prefix_fold,ModelsRuntimeInfo.fold_path,ModelsRuntimeInfo.config_name)
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

            if len(object_names)>0:
                if object_name not in object_names:
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

                        if print_info:
                            print("\n")
                            print("device name",device_name)
                            print("device id",device_id)
                            print("object name: ",object_name)
                            print("shape", shape)
                            # print("batch size(doing):   ",end="")

                        component_add_list={}
                        threads=[]

                        value = runtime_config[device_name][object_name][device_id][shapes_dimensionality][shape]
                        with open(os.path.join(ModelsRuntimeInfo.prefix_fold, value["file_path"]),"r") as f:
                            line = f.readline()
                            while line is not None and len(line)>0 :
                                batch_size = int(line.split(",")[0])
                                # if print_info:
                                    # pass
                                    # print("\b\b\b%-3d"%batch_size,end="")

                                # runtime = float(line.split(",")[1])

                                dshapes = ast.literal_eval(shape.replace("-1",str(batch_size)))

                                if not exist_in_dataset(model_name=object_name,shape=shape,batch_size=batch_size,dataset_name=SaveInfo.config_name,fold_path=SaveInfo.fold_path,prefix_dataset_name=prefix_dataset_name):
                                    thd = threading.Thread(target=analyze_model_component,args=(object_name,dshapes,batch_size,component_add_list,print_info),daemon=True)
                                    threads.append(thd)
                                    
                                    # flag,model_component_dict = deal_function(object_name,dshapes,shape=shape)

                                    # if not flag:
                                    #     # print("analyze model component error: ",error_info)
                                    #     return False

                                    if not TVM_INIT and print_info:
                                        TVM_INIT = True
                                        print("\n\noriginal device name",device_name)
                                        print("original device id",device_id)
                                        print("object name: ",object_name)
                                        print("shape", shape)
                                        # print("batch size(doing):   ",end="")

                                    # component_add_list[batch_size]=model_component_dict

                                    print("arrange batch_size=%d to thead: (name: %s, id: %d)."%(batch_size,thd.getName(),thd.ident))

                                    # add_model_component(model_name=object_name,shape=shape,batch_size=batch_size,component_dict=model_component_dict,prefix_dataset_name=prefix_dataset_name,dataset_name=SaveInfo.config_name,fold_path=SaveInfo.fold_path,auto_skip=SaveInfo.auto_skip)

                                line = f.readline()
                        
                        for t in threads:
                            t.start()
                            t.join()        # 同步

                        if print_info:
                            print("finish calculate all batch size.")
                            print("writing batch size:   ")
                            
                        for batch_size_k,component_dict_v in component_add_list:
                            print("\b\b\b%-3d"%batch_size)
                            add_model_component(model_name=object_name,shape=shape,batch_size=batch_size_k,component_dict=component_dict_v,prefix_dataset_name=prefix_dataset_name,dataset_name=SaveInfo.config_name,fold_path=SaveInfo.fold_path,auto_skip=SaveInfo.auto_skip)

    return True

# 待实现
def analyze_model_component(model_name,dshape,batch_size,component_add_list,print_info=True,dtype="float32"):
    try:
        device = tvm.cpu(0)
        target = "llvm"
        component_add_list[batch_size] = get_op_info(model_name,dshape[0],device=device,target=target)
        if print_info:
            print("%s finish work for batch_siz=%d."%(threading.current_thread().name,batch_size))
    except:
        traceback.print_exc()
        exit(-1)

def main():
    if analyze_models_by_runtime_json(analyze_model_component,object_names=["inception_v3","mobilenet","resnet-50","resnet3d-50","squeezenet_v1.1"],print_info=True):
        print("finish all works.")
    else:
        print("break off...")

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc()