from create_dataset.common import generate_datasets_with_one_dimensionality_changing
import tvm.relay as relay
import random
import create_dataset.test_code.op_test_code.config.common_args as common_args

def getRandomTuple(min_shape,max_shape):
    a = random.randint(min_shape,max_shape-1)
    b = random.randint(min_shape,max_shape-1)
    while a==b:
        b = random.randint(min_shape,max_shape-1)

    if a<b:
        return a,b
    else:
        return b,a

def calculate_op(dshape,dtype="float32"):
    x = relay.var("input_x", shape=dshape[0], dtype=dtype)

    shape_1 = getRandomTuple(0,dshape[0][1])
    shape_2 = getRandomTuple(0,dshape[0][2])
    shape_3 = getRandomTuple(0,dshape[0][3])

    f = relay.strided_slice(x,begin=(shape_1[0],shape_2[0],shape_3[0]),end=(shape_1[1],shape_2[1],shape_3[1]))

    return f

# 定义参数
function_dict = {"func":calculate_op, "name": "add"}

# 限定shape
count = 7
force_shape_relation=(None,)
shapes_dimensionality=((4,),(0,0))
range_min = ((-1,10,10,10),)
range_max = ((-1,100,100,100),)

generate_datasets_with_one_dimensionality_changing(device_parame_array=common_args.device_parame_array,count=count,shape_dimensionality=shapes_dimensionality,range_min=range_min,range_max=range_max,function_dict = function_dict,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,fold_path=common_args.fold_path,device_name=common_args.device_name,show_print=common_args.show_print)
