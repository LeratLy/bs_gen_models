import os

from src.analysis.latents import create_prototype_latents
from variables import DATA_DIR

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    create_prototype_latents(os.path.join(DATA_DIR, "analysis_data", "original_test"))
