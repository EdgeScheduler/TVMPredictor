from create_dataset.common import Device

min_shapes=1
max_shapes=128
sampling=1.0
dtype="float32"
cycle_times=3
min_repeat_ms=10
opt_level=0
fold_path="create_dataset/datasets_models/"
show_print=True
isModel = True
device_name="dell04"
config_name = "dataset_"+device_name+".json"

# device_parame_array = [Device.device_params_CPU,Device.device_params_GPU0]
device_parame_array = [Device.device_params_GPU1,]