from create_dataset.common import generate_datasets_with_one_dimensionality_changing,uniform_sampling,generate_dataset_with_one_dimensionality_changing
import tvm.relay as relay
import create_dataset.test_code.model_test_code.config.common_args as common_args
from create_dataset.test_code.model_create.get_tvm_model import create

def calculate_model(dshape,dtype="float32"):
    return create("NN_1",dshape=dshape,middle_size_1=256,middle_size_2=64,out_size=10)

# 定义参数
function_dict = {"func":calculate_model, "name": "NN_1"}


force_shape_relation=(None,)
shapes_dimensionality=((3,),(0,0))

count = 1
range_min = [[-1,1,1],]
range_max = [[-1,101,101],]
for a in uniform_sampling(1,100,0.1):
    for b in uniform_sampling(1,100,0.1):
        range_min[0][1]=a
        range_max[0][1]=a
        range_min[0][2]=b
        range_max[0][2]=b

        generate_datasets_with_one_dimensionality_changing(device_parame_array=common_args.device_parame_array,count=count,shape_dimensionality=shapes_dimensionality,range_min=range_min,range_max=range_max,function_dict = function_dict,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,fold_path=common_args.fold_path,device_name=common_args.device_name,show_print=common_args.show_print,isModule=common_args.isModel)
        count+=1