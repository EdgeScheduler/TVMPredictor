from multiprocessing.context import Process
import os
import json
import ast
from TVMProfiler.model_src.analyze_componnet import get_op_info
import tvm
import traceback
import multiprocessing
from multiprocessing import Lock, Pool

class SaveInfo:
    fold_path="Datasets/TVM/models_component/"
    config_name = "dataset.json"
    pre_prefix_config_name = "__"
    auto_skip = True

class ModelsRuntimeInfo:
    prefix_fold = "Datasets/"
    fold_path="TVM/datasets_models/"
    config_name = "dataset_auto.json"

def call_back(component_dict):
    return component_dict

def error_call_back(obj):
    print("---error: ",str(obj))
    traceback.print_exc()
    return {"error",str(obj)}


def init_lock(lk):
    global lock
    lock = lk

def analyze_models_by_runtime_json(deal_function,object_names=[],prefix_dataset_name="",print_info=True):
    lock = multiprocessing.Lock()

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
                    
                    pool = Pool(processes=multiprocessing.cpu_count(),initializer=init_lock, initargs=(lock,))
                    for shape in runtime_config[device_name][object_name][device_id][shapes_dimensionality].keys():
                        if shape=="count":
                            continue

                        batch_sizes=[]
                        with open(os.path.join(ModelsRuntimeInfo.prefix_fold, runtime_config[device_name][object_name][device_id][shapes_dimensionality][shape]["file_path"]),"r") as f:
                            line = f.readline()
                            while line is not None and len(line)>0 :
                                batch_sizes.append(int(line.split(",")[0]))
                                line = f.readline()
                    
                        pool.apply_async(func=deal_function,args=(SaveInfo.pre_prefix_config_name,device_name,object_name,device_id,shape,batch_sizes,SaveInfo.fold_path,True,),error_callback=error_call_back)

                    pool.close()
                    pool.join()

                    # 合并
                    if os.system("./home/yutian/TVMPredictor/Datasets/TVM/models_component/merge.sh")==0:
                        print("json config merge finished.")
                    else:
                        print("fail to merge json config.")
    return True

def run_single_shape(pre_prefix_config_name,device_name,object_name,device_id,shape,batch_sizes,fold_path,print_info=True):
    if print_info:
        print("\n")
        print("original device name",device_name)
        print("original device id",device_id)
        print("object name: ",object_name)
        print("shape", shape)

    shell_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"run_shell.py")
    dataset_name = pre_prefix_config_name+str(object_name)+"__"+str(shape).replace(" ","")+".json"
    # shell_str = "python3 -u %s -n %s -s %s -b %s -p %s -j %s"%(shell_path,object_name,str(shape).replace(" ",""),str(batch_sizes).replace(" ",""),fold_path,dataset_name)+ " >output/"+pre_prefix_config_name+str(object_name)+"_"+str(shape).replace(" ","")+"_output.txt"
    shell_str = "python3 -u %s -n %s -s %s -b %s -p %s -j %s"%(shell_path,object_name,str(shape).replace(" ",""),str(batch_sizes).replace(" ",""),fold_path,dataset_name)+ " >>output/"+"output.txt"
    
    shell_str = shell_str.replace("(","[").replace(")","]")
    # print("run shell:",shell_str)
    if os.system(shell_str)==0:
        if print_info:
            print("finish calculate all batch size for shape=%s."%shape)
    else:
        print("error when deal with:")
        print("original device name",device_name)
        print("original device id",device_id)
        print("object name: ",object_name)
        print("shape", shape)

def main():
    print("<%d cores in your device.>"%multiprocessing.cpu_count())
    if analyze_models_by_runtime_json(run_single_shape,object_names=["mobilenet","resnet-50","resnet3d-50","squeezenet_v1.1"],print_info=True):
        print("finish all works.")
    else:
        print("break off...")

if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc()