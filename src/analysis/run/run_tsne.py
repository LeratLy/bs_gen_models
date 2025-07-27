import argparse
import os.path

from src.analysis.latents import run_tsne
from src.analysis.utils import perform_for_all_folders
from variables import DATA_DIR


def setup_arg_parser():
    """
    Parse main arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=str, default="4", help="Number of threads for OMP/MKL/NUMEXPR")
    parser.add_argument("--base_path", type=str, default="raw_data",
                        help="Sub-path in DATA_DIR to data directory (containing plain numpy files) and features.pkl")
    parser.add_argument("--data_folder", type=str, default=None,
                        help="Sub-path in base_bath to data directory containing plain data (containing samples.npz, features.pkl)")
    parser.add_argument("--perplexities", type=str, default="30",
                        help="Perplexities to run code for (should be comma seperated, e.g 10,30,50")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_arg_parser()

    if args.num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    base_path = os.path.join(DATA_DIR, str(args.base_path))
    kwargs = {
        "perplexities": str(args.perplexities).split(","),
    }
    if args.data_folder is not None:
        run_tsne(os.path.join(base_path, args.data_folder), **kwargs)
    else:
        perform_for_all_folders(base_path, run_tsne, kwargs)

