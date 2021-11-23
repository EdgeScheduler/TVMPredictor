from create_dataset.fix_json.fix_count_files import fix_count_files
from create_dataset.fix_json.delete_unexist_config import delete_unexist_config
import create_dataset.test_code.model_test_code.config.common_args as common_args
import os

delete_unexist_config(os.path.join(common_args.prefix_path,common_args.fold_path,common_args.config_name),prifex_fold=common_args.prefix_path)
fix_count_files(os.path.join(common_args.prefix_path,common_args.fold_path,common_args.config_name))