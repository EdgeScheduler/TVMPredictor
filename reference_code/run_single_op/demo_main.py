# 运行单个算子: https://blog.csdn.net/weixin_39713833/article/details/111389194

import tvm
# from tvm.relay import transform
import tvm.relay as relay
import numpy as np
from tvm.contrib import graph_runtime
# from TVMProfiler.relayIR.relay_graph import construct_op_graph, profile_resource_usage

def batch_norm_infer(data, gamma=None, beta=None, moving_mean=None, moving_var=None, **kwargs):
    name = kwargs.get("name")  
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data, gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var, **kwargs)[0]

def conv2d(data, weight=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)

# def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), epsilon=1e-5):
#     # return conv2d(
#     #      data=data,
#     #      channels=channels,
#     #      kernel_size=kernel_size,
#     #      strides=strides,
#     #      padding=padding,
#     #      data_layout='NCHW',
#     #      name=name+'_conv')
#     return batch_norm_infer(data=data, epsilon=epsilon, name=name + '_bn')
#     # act = relay.nn.relu(data=bn)
#     # return act

kernel_shape = (32, 3, 3, 3)
data_shape = (32, 112, 112)
dtype = "float32"
data = relay.var("data", shape=data_shape, dtype=dtype)
act = batch_norm_infer(data=data, epsilon=1e-5, name="graph_bn")
func = relay.Function(relay.analysis.free_vars(act),act)

mod = tvm.ir.IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)
shape_dict = {
    v.name_hint : v.checked_type for v in mod["main"].params}

np.random.seed(0)
params = {}
for k, v in shape_dict.items():
    if k == "data":
        continue
    init_value = np.random.uniform(-1, 1, v.concrete_shape).astype(v.dtype)
    params[k] = tvm.nd.array(init_value, device=tvm.cpu(0))

target = "llvm"
device = tvm.cpu(0)
# print("Relay module function:\n", mod.astext(show_meta_data=False))
# print("TVM parameters:\n", params.keys())

with relay.build_config(opt_level=3):
    graph, lib, params2 = relay.build(mod, target, params=params)

# print("TVM graph:\n", graph)
# print("TVM parameters:\n", params.keys())
# print("TVM compiled target function:\n", lib.get_source())
print(mod)
module = graph_runtime.create(graph, lib, device)
data_tvm = np.random.uniform(1000, 50, size=data_shape).astype(dtype)
# batch_norm_input = np.random.uniform(-1, 1, size=(1, 32, 112, 112)).astype(dtype)
module.set_input('data', data_tvm)
module.set_input(**params)
module.run()
output = module.get_output(0)
# construct_op_graph(mod)
# profile_resource_usage(params,data_tvm, device=device, target = target)


entrance_tuple = mod.functions.items()[0]
print("-------\n",mod.functions.items(),"\n")
main_function = entrance_tuple[1]

temp_body2 = tvm.relay.Call(main_function.body.tuple_value.op, main_function.body.tuple_value.args, attrs=main_function.body.tuple_value.attrs, type_args=main_function.body.tuple_value.type_args)
temp_body = tvm.relay.expr.TupleGetItem(temp_body2,0)
call_function = tvm.relay.Function(relay.analysis.free_vars(temp_body),temp_body)
call_functions = {"main": call_function}
call_ir_module = tvm.ir.IRModule(functions=call_functions)
with tvm.transform.PassContext(opt_level=1):
    call_interpreter = relay.build_module.create_executor("graph", call_ir_module, device, target)

print("--call_ir_module:\n",call_ir_module,"\n")
input_args = []
input_args.append(data_tvm)
print("--params.keys():\n",params.keys(),"\n")
res = call_interpreter.evaluate()(*input_args, **params)

# print("--output:\n",output,"\n")
# print("--res:\n",res,"\n")