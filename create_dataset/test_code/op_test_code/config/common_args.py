from sys import prefix
from create_dataset.common import Device

min_shapes=10
max_shapes=100
sampling=0.1
dtype="float32"
cycle_times=20
min_repeat_ms=30
opt_level=0
prefix_path="Datasets/"
fold_path="TVM/datasets_huyi/"
show_print=True

device_name="dell04"
# device_parame_array = [Device.device_params_CPU,Device.device_params_GPU0]
device_parame_array = [Device.device_params_GPU0,]