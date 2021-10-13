from create_dataset.common import Device

min_shapes=1
max_shapes=100
sampling=1.0
dtype="float32"
cycle_times=20
min_repeat_ms=30
opt_level=0
fold_path="create_dataset/datasets/"
show_print=True

device_name="dell03"
# device_parame_array = [Device.device_params_CPU,Device.device_params_GPU0]
device_parame_array = [Device.device_params_GPU0,]