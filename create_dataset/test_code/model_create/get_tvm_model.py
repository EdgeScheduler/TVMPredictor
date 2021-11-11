from create_dataset.test_code.model_create import *
import onnx
import tvm
import os
from create_dataset.common import generate_datasets_with_one_dimensionality_changing,Device

def create(model_name,dshape,onnx_name="onnx_tmp.onnx", dtype="float32", **args):
   
    '''
    test add-op in one kind of shape.

    Parameters
    ----------

    exp:
    * GPU: target = "cuda", device = tvm.cuda(0)
    * CPU: target = "llvm", device=tvm.cpu(0)
    '''
    # 生成模型
    onnx_name = eval(model_name+".create_onnx")(dshape,onnx_name,**args)

    onnx_model = onnx.load(onnx_name)

    shape_dict = None
    if model_name=="NN_1":
        shape_dict = {'input': dshape[0]}
    elif model_name=="CNN":
        shape_dict = {'input': dshape[0]}
    else:
        print("unknow model: function<create> in get_tvm_model.py")
        return None

    # 获取模型
    mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape_dict)

    return mod