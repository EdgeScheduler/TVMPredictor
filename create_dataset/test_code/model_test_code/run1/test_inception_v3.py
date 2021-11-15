from create_dataset.test_code.model_test_code.operation_statistics.cnn_workload_generator import get_network
from create_dataset.common import generate_datasets_with_one_dimensionality_changing,uniform_sampling,generate_dataset_with_one_dimensionality_changing
import tvm.relay as relay
import create_dataset.test_code.model_test_code.config.common_args as common_args

def calculate_model(dshape,dtype="float32"):
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''

    return get_network("inception_v3", input_shape=dshape[0], dtype=dtype)[0]

# 定义参数
function_dict = {"func":calculate_model, "name": "inception_v3"}

force_shape_relation=(None,)
shapes_dimensionality=((4,),(0,0))

count = 1
range_min = [[-1,3,16,16],]
range_max = [[-1,3,256,256],]
for a in uniform_sampling(16,256,0.0625):
    for b in uniform_sampling(16,256,0.0625):
        range_min[0][2]=a
        range_max[0][2]=a
        range_min[0][3]=b
        range_max[0][3]=b

        generate_datasets_with_one_dimensionality_changing(device_parame_array=common_args.device_parame_array,count=count,shape_dimensionality=shapes_dimensionality,range_min=range_min,range_max=range_max,function_dict = function_dict,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,fold_path=common_args.fold_path,device_name=common_args.device_name,show_print=common_args.show_print,isModule=common_args.isModel,dataset_config_name=common_args.config_name)
        count+=1