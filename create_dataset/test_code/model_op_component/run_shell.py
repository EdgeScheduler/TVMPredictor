from multiprocessing.context import Process
import os
import json
import ast
from TVMProfiler.model_src.analyze_componnet import get_op_info
import tvm
import traceback
import multiprocessing
from multiprocessing import Pool
from optparse import OptionParser

def run_in_terminal():
    parser = OptionParser(usage="define the args when analyze the model")
    parser.add_option("-n", "--name", action="store",dest="model_name",default="",help="model name")
    parser.add_option("-s", "--shape", action="store",dest="shape",default="",help="the shape of input<python-tuple/list>, batch size should be -1.")
    parser.add_option("-b", "--batch_size", action="store",dest="batch_range",default="",help="range of batch size<python-tuple/list>.")
    parser.add_option("-p", "--path", action="store",dest="json_path",default="",help="json store fold path.")
    parser.add_option("-j", "--json_name", action="store",dest="json_name",default="",help="name of json.")
    (options, args) = parser.parse_args()

    if options.model_name=="" or options.shape=="" or options.batch_range=="" or options.json_path=="" or options.json_name=="":
        print("bad input:")
        print("model name:",options.model_name)
        print("shape:",options.shape)
        print("batch range:",options.batch_range)
        print("json fold:",options.json_path)
        print("json name:",options.json_name)
        return

    options_model_name = str(options.model_name).replace("[","(").replace("]",")").replace(" ","")
    options_shape = str(options.shape).replace("[","(").replace("]",")").replace(" ","")
    options_batch_range = str(options.batch_range).replace("[","(").replace("]",")").replace(" ","")
    options_json_path = str(options.json_path).replace("[","(").replace("]",")").replace(" ","")
    options_json_name = str(options.json_name).replace("[","(").replace("]",")").replace(" ","")

    print("process-%d is running for model=%s, shape=%s."%(os.getpid(),options_model_name,options_shape))

    batch_sizes = ast.literal_eval(options_batch_range)

    if not os.path.exists(options_json_path):
        os.makedirs(options_json_path)
    
    json_path = os.path.join(options_json_path,options_json_name)
    # for batch_size in batch_sizes:
    for batch_size in [1,2]:
        dshape = ast.literal_eval(options_shape.replace("-1",str(batch_size)))
        device = tvm.cpu(0)
        target = "llvm"

        record_dict = {}
        if os.path.exists(json_path):
            with open(json_path,'r') as f:
                record_dict = json.load(f)

        if str(batch_size)not in record_dict.keys():
            record_dict[str(batch_size)]=get_op_info(options_model_name,dshape[0],device=device,target=target)

            with open(json_path,"w") as f:
                json.dump(record_dict,fp=f,indent=4,separators=(',', ': '),sort_keys=True)

        print("process-%d done for model=%s, shape=%s, batch_size=%d."%(os.getpid(),options_model_name,options_shape,batch_size))

    print("process-%d finished exist."%os.getpid())

if __name__ == "__main__":
    run_in_terminal()