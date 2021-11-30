from create_dataset.common import Device, test_op_time
import tvm.relay as relay
import create_dataset.test_code.op_test_code.config.common_args as common_args
import os
import json
import ast
import copy

from tvm.runtime.ndarray import device

class Param:
    cycle_times=20
    min_repeat_ms=30
    opt_level=0
    device = Device.device_params_GPU0
    save_fold = "../../../../Datasets/TVM/models_component/"
    json_name="dataset_time_analyze.json"

def GetFunction(op_name,args_list,params_list,dtype="float32"):
    # print("---op-name=",op_name,"args=",args_list,"params=",params_list)
    f = None
    if op_name=="split":
        print("op: split need to be deal in code.")
        pass
    elif op_name=="strided_slice":
        print("op: strided_slice need to be deal in code.")
        pass
    elif op_name=="concatenate":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=args_list[1], dtype=dtype)
        f = relay.concatenate(x, y)
    elif op_name=="nn.dense":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=params_list[0], dtype=dtype)
        f = relay.nn.dense(data=x,weight=y,out_dtype=dtype)
    elif op_name=="add":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=args_list[1], dtype=dtype)
        f = relay.add(x, y)
    elif op_name=="sigmoid":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        f = relay.sigmoid(x)
    elif op_name=="tanh":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        f = relay.tanh(x)
    elif op_name=="multiply":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=args_list[1], dtype=dtype)
        f = relay.multiply(x, y)
    elif op_name=="subtract":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=args_list[1], dtype=dtype)
        f = relay.subtract(x, y)
    elif op_name=="nn.conv2d":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=params_list[0], dtype=dtype)
        f = relay.nn.conv2d(x, y,padding=(params_list[0][2]-1,params_list[0][3]-1))
    elif op_name=="nn.bias_add":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        y = relay.var("input_y", shape=params_list[0], dtype=dtype)
        f = relay.nn.bias_add(x, y)
    elif op_name=="nn.relu":
        x = relay.var("input_x", shape=args_list[0], dtype=dtype)
        f = relay.nn.relu(x)
    else:
        return None
    return f

def GetRealRuntime(model_name,model_input_shape,batch_size,device_name="dell04",device_id_list=["0","1"],changing_shape="((4,), (0, 0))",pre_fold="Datasets/"):
    datas = {}
    batch_size = int(batch_size)
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../Datasets/TVM/datasets_models/dataset_auto.json"),"r") as f:
            datas = json.load(f)
        # print(datas[device_name][model_name]["0"][changing_shape]["((-1,3,192,208),)"].keys())
        # exit(1)
        file_name = None
        for device_id in device_id_list:
            if device_id in datas[device_name][model_name].keys() and changing_shape in datas[device_name][model_name][device_id].keys() and model_input_shape in datas[device_name][model_name][device_id][changing_shape].keys():
                file_name = datas[device_name][model_name][device_id][changing_shape][model_input_shape]["file_path"]
                break

        if file_name is None:
            raise Exception('expect shape is not in database.')

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../",pre_fold,file_name)) as f:
            line = f.readline()
            while line is not None and len(line)>0 :
                batch = int(line.split(",")[0])
                if batch==batch_size:
                    return float(line.split(",")[1])
                elif batch>batch_size:
                    break
                else:
                    line=f.readline()
        raise Exception('batch size for this shape is not in database.')
    except Exception as ex:
        print("error info: ",str(ex))
        print("maybe loss data: model_name=%s,model_input_shape=%s,batch_size=%s"%(model_name,model_input_shape,batch_size))
        return -1.0
def main():
    allow_loss_rate = 0.05

    program_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(program_path,"../../../../Datasets/TVM/models_component/dataset_time_analyze.json")
    datas = {}
    with open(json_path,"r") as f:
        datas = json.load(f)

    total_accuracy = 0.0
    for model_name in datas.keys():
        if model_name in ["accuracy","loss","allow_loss_rate"]:
            continue
        model_total_accuracy = 0.0
        for model_input_shape in datas[model_name].keys():
            if model_input_shape in ["accuracy","loss"]:
                continue
            model_input_total_accuracy = 0.0
            for batch_size in datas[model_name][model_input_shape].keys():
                if batch_size in ["accuracy","loss"]:
                    continue
                batch_total_runtime = datas[model_name][model_input_shape][batch_size]["runtimes"]
                real_runtime = datas[model_name][model_input_shape][batch_size]["real_runtimes"]
                datas[model_name][model_input_shape][batch_size]["loss_rate"] = abs(batch_total_runtime-real_runtime)/real_runtime

                # 计算准确率
                if abs(batch_total_runtime-real_runtime)/real_runtime < allow_loss_rate:
                    model_input_total_accuracy += abs(batch_total_runtime-real_runtime)/real_runtime

            avg_model_shape_accuracy = model_input_total_accuracy/len(datas[model_name][model_input_shape].keys())
            datas[model_name][model_input_shape]["accuracy"] = avg_model_shape_accuracy
            model_total_accuracy += avg_model_shape_accuracy
        avg_model_accuracy = model_total_accuracy/len(datas[model_name].keys())
        print("model avg pre-accuracy for %s: %.2f%%"%(model_name,avg_model_accuracy*100))

        datas[model_name]["accuracy"] = avg_model_accuracy
        total_accuracy += avg_model_accuracy
    datas["accuracy"] = total_accuracy/len(datas.keys())
    datas["allow_loss_rate"] = allow_loss_rate
    print("total model avg pre-accuracy: %.2f%%"%(avg_model_accuracy*100))

    with open(os.path.join(program_path,Param.save_fold,"_"+str(allow_loss_rate)+"_"+Param.json_name),"w") as f:
        json.dump(datas,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

if __name__=='__main__':
    main()