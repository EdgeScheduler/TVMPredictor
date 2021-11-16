print("there is some bug in tvm.replay.split, skip this test...")
exit(0)

from create_dataset.common import generate_datasets_with_one_dimensionality_changing
import tvm.relay as relay
import create_dataset.test_code.op_test_code.config.common_args as common_args

def calculate_op(dshape,dtype="float32"):
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)
    # y = relay.var("input_y", shape=dshape[1], dtype=dtype)
    print(dshape)
    f = relay.split(x,indices_or_sections=dshape[1],axis=1)

    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "split"}

# 限定shape
count = 7
force_shape_relation=(None,(lambda x,y:y,))
shapes_dimensionality=((2,1),(0,0))
range_min = ((-1,1),(1,))
range_max = ((-1,100),(100,))

generate_datasets_with_one_dimensionality_changing(device_parame_array=common_args.device_parame_array,count=count,shape_dimensionality=shapes_dimensionality,range_min=range_min,range_max=range_max,function_dict = function_dict,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,prefix_path=common_args.prefix_path,fold_path=common_args.fold_path,device_name=common_args.device_name,show_print=common_args.show_print)
