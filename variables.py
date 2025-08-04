import os
import numpy as np

from src._types import Mode
from src.utils.helpers import path_resolver

# Paths used within the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(os.path.join(ROOT_DIR, "reports"), "figures") # Figures will be saved to this directory
DATA_DIR = os.path.join(ROOT_DIR, "data") # Data is searched within this directory
MODEL_DIR = os.path.join(ROOT_DIR, "models") # Trained models will be loaded from this directory
ANALYSIS_DATA_DIR = os.path.join(DATA_DIR, "analysis") # Data for analysis will be loaded from this directory

DEVICE = "cpu"

META_MS = "MSData_noNMO.xlsx"
META_HC = "HCData.xlsx"

HC_FOLDER = "Dataset002_DFGFinetuned_3d_fullres_full_new_hc"
MS_FOLDER = "Dataset002_DFGFinetuned_3d_fullres_full_ms"

# paths to datasets
data_paths = {
    'chP3D_tune_5_classes_70': path_resolver(4, DATA_DIR),
    'chP3D_tune_full_5_classes': path_resolver(4, DATA_DIR),
}

split_csvs = {
    # split csv for testing
    'chP3D_tune_5_classes_70': {
        Mode.train: path_resolver(4, os.path.join(DATA_DIR, "train_split_tune_5_classes_70.csv")),
        Mode.val: path_resolver(4, os.path.join(DATA_DIR, "val_split_tune_5_classes_70.csv")),
        Mode.test: path_resolver(4, os.path.join(DATA_DIR, "test_split_tune_5_classes_70.csv"))
    },
    'chP3D_tune_full_5_classes': {
        Mode.train: path_resolver(4, os.path.join(DATA_DIR, "train_full_data_split_5_classes.csv")),
    }
}

# --------------------------- Label name mappings ----------------------------------
MS_TYPES = {
    "HC": 0,
    "CIS": 1,
    "RRMS": 3,
    "PPMS": 2,
    "SPMS": 4
}
LABEL_TO_NAME = {
    0: "HC",
    1: "CIS",
    2: "PPMS",
    3: "RRMS",
    4: "SPMS",
}
LABEL_TO_MAIN_NAME = {
    0: "MS",
    1: "HC",
}
MAIN_NAME_TO_LABEL = {
    "HC": 1,
    "MS": 0,
}
MS_MAIN_TYPE = {
    "CIS": 0,
    "PPMS": 0,
    "RRMS": 0,
    "SPMS": 0,
    "HC": 1
}
MS_TYPES_TO_MAIN_TYPE = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    0: 1
}
MS_TYPES_FILTERED = {
    "HC": 0,
    "CIS": 1,
    "RRMS": 2,
}
MS_TYPES_NAMES = np.array(list(LABEL_TO_NAME.values()))
MS_MAIN_TYPES_NAMES = np.array(list(LABEL_TO_MAIN_NAME.values()))

# ----------------------------------------------------------------------------------

# --------------------------- colors -----------------------------------------------
MS_TYPE_MAIN_COLORS = {
    0: "darkorange",
    1: "seagreen"
}
MS_TYPE_COLORS = {
    0: "seagreen",
    1: "royalblue",
    2: "goldenrod",
    3: "firebrick",
    4: "mediumpurple"
}
MS_TYPE_MAIN_COLORS_ARRAY = np.array(list(MS_TYPE_MAIN_COLORS.values()))
MS_TYPE_COLORS_ARRAY = np.array(list(MS_TYPE_COLORS.values()))
# ----------------------------------------------------------------------------------


# --------------------------- Edge colors ------------------------------------------
MS_TYPE_MAIN_EDGECOLORS = {
    0: "chocolate",
    1: "darkgreen"
}
MS_TYPE_EDGECOLORS = {
    0: "darkgreen",
    1: "midnightblue",
    2: "darkgoldenrod",
    3: "darkred",
    4: "indigo"
}
MS_TYPE_EDGECOLORS_ARRAY = np.array(list(MS_TYPE_EDGECOLORS.values()))
MS_TYPE_MAIN_EDGECOLORS_ARRAY = np.array(list(MS_TYPE_MAIN_EDGECOLORS.values()))
# ----------------------------------------------------------------------------------

# ---------------------------Markers -----------------------------------------------
MS_TYPE_MAIN_MARKERS = {
    0: "o",
    1: "^"
}
MS_TYPE_MARKERS = {
    0: "^",
    1: "o",
    2: "o",
    3: "o",
    4: "o"
}
MS_TYPE_MARKERS_ARRAY = np.array(list(MS_TYPE_MARKERS.values()))
MS_TYPE_MAIN_MARKERS_ARRAY = np.array(list(MS_TYPE_MAIN_MARKERS.values()))
# ----------------------------------------------------------------------------------

# --------------------------- Type infos- ------------------------------------------
MS_MAIN_TYPE_DETAILS = {
    "names": MS_MAIN_TYPES_NAMES,
    "colors": MS_TYPE_MAIN_COLORS_ARRAY,
    "edgecolors": MS_TYPE_MAIN_EDGECOLORS_ARRAY,
    "markers": MS_TYPE_MAIN_MARKERS_ARRAY,
}
MS_TYPE_DETAILS = {
    "names": MS_TYPES_NAMES,
    "colors": MS_TYPE_COLORS_ARRAY,
    "edgecolors": MS_TYPE_EDGECOLORS_ARRAY,
    "markers": MS_TYPE_MARKERS_ARRAY,
}
# ----------------------------------------------------------------------------------
