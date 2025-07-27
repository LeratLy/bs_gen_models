# ----------------------------------------------------------------------------
import argparse
import os

from src.analysis.samples import original_to_npz, sample, infer
from variables import DATA_DIR


def setup_arg_parser():
    """
    Parse main arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=str, default="4", help="Number of threads for OMP/MKL/NUMEXPR")
    parser.add_argument("--cuda_device", type=str, default=None, help="CUDA device ID (e.g., '4,5')")
    parser.add_argument("--no_original", action="store_true", help="Whether to save original 'samples' to file")
    parser.add_argument("--no_sampled", action="store_true", help="Whether to save new samples ones to file")
    parser.add_argument("--no-infer", action="store_true", help="Whether to save inferred latents to file")
    parser.add_argument("--device", type=str, default="cpu", help="Which device to use for model")
    parser.add_argument("--model_name", required=True, type=str, default=None, help="Type of model (e.g 'cvae', 'bdae')")
    parser.add_argument("--checkpoint_dir", required=True, type=str, default=None,
                        help="Sub-path in DATA_DIR to checkpoint file")
    parser.add_argument("--base_path", type=str, default="raw_data",
                        help="Sub-path in DATA_DIR to data directory (containing plain numpy files) and to story features.pkl")
    return parser.parse_args()


def main():
    args = setup_arg_parser()

    if args.cuda_device is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if args.num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    model_name = args.model_name
    if model_name == "cvae":
        checkpoint_dir = os.path.join(
            DATA_DIR,
            "last_tuning_round",
            "checkpoints_last_cvae",
            "FINAL_cvae_minlr0.0001dropout0_ch16_num_layers4_num_target_emb_channels1_base_20250629_222215_best"
        )
    else:
        raise NotImplementedError()

    base_path = os.path.join(DATA_DIR, str(args.base_path))
    if not args.no_sampled:
        for num, name in zip([[388, 36], [84, 12]], ["sampled_train", "sampled"]):
            sampled_path = os.path.join(base_path, f"{model_name}_{name}")
            sample(sampled_path, model_name, checkpoint_dir, num)
    if not args.no_original:
        original_to_npz(base_path)
    if not args.no_infer:
        infer(base_path, model_name, checkpoint_dir, args.device)


if __name__ == "__main__":
    main()
