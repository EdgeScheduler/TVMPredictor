import relayIR.onnx_profiler as onnx_profiler
import op_detectm
import tvm
import numpy as np
import tvm.relay
from create_dataset.test_code.model_op_component.operation_statistics.cnn_workload_generator import get_network

# 从代码生成
def analyze_component(model_name,dshapes, device = tvm.cuda(0), target = "cuda"):
    try:
        datas = [np.random.uniform(-10, 10, shape).astype("float32") for shape in dshapes]

        mod, params, input_shape, output_shape = get_network(model_name, dshapes[0])
        with tvm.transform.PassContext(opt_level=1):
            intrp = tvm.relay.build_module.create_executor("graph", mod, device = device, target = target)

        onnx_profiler.run_relay_mod([np.random.uniform(-10, 10, input_shape).astype("float32")], intrp, params)
        op_detectm.construct_op_graph(mod)
        return True,op_detectm.profile_resource_usage(params, {"input":datas[0]},["input"], device = device, target = target)
    except:
        return False,{}

