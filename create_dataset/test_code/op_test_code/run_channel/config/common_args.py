from sys import prefix
from create_dataset.common import Device

min_shapes=1
max_shapes=200
sampling=1.0
dtype="float32"
cycle_times=20
min_repeat_ms=30
opt_level=0
prefix_path="Datasets/"
fold_path="TVM/datasets_channel/"
show_print=True
config_name= "dataset.json"

device_name="dell04"
device_parame_array = [Device.device_params_GPU1,]