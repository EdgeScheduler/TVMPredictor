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

    return get_network("resnet-50", input_shape=dshape[0], dtype=dtype)[0]

# 定义参数
function_dict = {"func":calculate_model, "name": "resnet-50"}

force_shape_relation=(None,)
shapes_dimensionality=((4,),(0,0))

shapes = [[-1,3,0,0],]
for a in uniform_sampling(16,256,0.0625):
    for b in uniform_sampling(16,256,0.0625):
        shapes[0][2]=a
        shapes[0][3]=b

        for device_parame in common_args.device_parame_array:
            generate_dataset_with_one_dimensionality_changing(function_dict=function_dict,shapes=shapes,min_shapes=common_args.min_shapes,max_shapes=common_args.max_shapes,sampling=common_args.sampling,force_shape_relation=force_shape_relation,device_parames=device_parame,dtype=common_args.dtype,cycle_times=common_args.cycle_times,min_repeat_ms=common_args.min_repeat_ms,opt_level=common_args.opt_level,prefix_path=common_args.prefix_path,fold_path=common_args.fold_path,device_name=common_args.device_name,dataset_config_name=common_args.config_name,show_print=common_args.show_print,isModule=common_args.isModel)