import argparse
import os

from src.analysis.shape_metrics import geometrics_for_folder
from src.analysis.utils import perform_for_all_folders
from variables import DATA_DIR


def setup_arg_parser():
    """
    Parse main arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="raw_data",
                        help="Sub-path in DATA_DIR to data directory containing plain data (containing samples.npz, features.pkl) or multiple directories containing plain data")
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Sub-path in base_bath to data directory containing plain data (containing samples.npz, features.pkl)")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_arg_parser()
    base_path = os.path.join(DATA_DIR, str(args.base_path))
    if args.data_folder is not None:
        base_path = os.path.join(base_path, str(args.data_folder))
        geometrics_for_folder(base_path)
    else:
        perform_for_all_folders(base_path, geometrics_for_folder)

