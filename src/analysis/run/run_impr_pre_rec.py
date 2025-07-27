# ----------------------------------------------------------------------------
import argparse
import os

from src.analysis.evaluation.impr_pre_rec import calc_impr_prec_rec_multiple
from src.analysis.latents import infer_latents_multiple
from src.analysis.utils import load_model_with_checkpoint
from variables import DATA_DIR


def setup_arg_parser():
    """
    Parse main arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=str, default="4", help="Number of threads for OMP/MKL/NUMEXPR")
    parser.add_argument("--infer_latents", action="store_true", help="Enable latent inference before calculating metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Device used for inference and metric calculation (e.g 'cpu' or 'cuda')")
    parser.add_argument("--cuda_device", type=str, default=None, help="CUDA device ID (e.g., '4,5')")
    parser.add_argument("--model_name", type=str, required=True, default=None, help="Model name (e.g., 'cvae')")
    return parser.parse_args()

def main():
    args = setup_arg_parser()

    if args.cuda_device is not None: os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if args.num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    model_name = args.model_name
    is_infer_latents = args.infer_latents

    print(f"Using model: {model_name}")
    print(f"Infer latents: {is_infer_latents}")
    print(f"CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Num threads: {args.num_threads}")

    base_folder = os.path.join(DATA_DIR, "raw_data")
    original_folders = ["original_test", "original_val", "original_train"]
    sampled_folders = [f"{model_name}_sampled", f"{model_name}_sampled", f"{model_name}_sampled_train"]
    data_folders = list(set(original_folders + sampled_folders))

    model = load_model_with_checkpoint(model_name, os.path.join(
                DATA_DIR,
                "last_tuning_round",
                "checkpoints_last_cvae",
                "FINAL_cvae_minlr0.0001dropout0_ch16_num_layers4_num_target_emb_channels1_base_20250629_222215_best"
            ))
    if is_infer_latents:
        infer_latents_multiple(model, data_folders, base_folder)
    calc_impr_prec_rec_multiple(original_folders, sampled_folders, base_folder, device=args.device)

if __name__ == "__main__":
    main()
