from create_dataset.common import generate_datasets_with_one_dimensionality_changing,uniform_sampling,generate_dataset_with_one_dimensionality_changing
import tvm.relay as relay
import create_dataset.test_code.op_test_code.config.common_args as common_args

def calculate_op(dshape,dtype="float32"):
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    f = relay.nn.relu(x)
    
    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "nn.relu"}

count = 7
force_shape_relation=(None,)
shapes_dimensionality=((4,),(0,0))
# range_min = ((-1,1,1),)
# range_max = ((-1,100,100),)

# generate_datasets_with_one_dimensionality_changing(device_parame_array=common_args.device_parame_array,count=count,shape_dimensionality=shapes_dimensionality,range_min=range_min,range_max=range_max,function_dict = function_dict,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,prefix_path=common_args.prefix_path,fold_path=common_args.fold_path,device_name=common_args.device_name,show_print=common_args.show_print)

shapes = [[-1,0,0,0],[0,0,0,0]]
for a in uniform_sampling(1,150,0.0666):
    for b in uniform_sampling(1,150,0.0666):
        for c in uniform_sampling(1,150,0.0666):
            shapes[0][1]=a
            shapes[0][2]=b
            shapes[0][3]=c
            
            for device_parame in common_args.device_parame_array:
                generate_dataset_with_one_dimensionality_changing(function_dict=function_dict,shapes=shapes,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,device_parames=device_parame,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,prefix_path=common_args.prefix_path,fold_path=common_args.fold_path,device_name=common_args.device_name,dataset_config_name=common_args.config_name,show_print=common_args.show_print)
