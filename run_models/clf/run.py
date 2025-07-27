import argparse
import os

from run_models.clf_templates import get_chP96_clf_5cond_conf
from src.models.trainer import Trainer
from variables import DATA_DIR


def train(conf):
    conf.__post_init__()
    conf.name += "5_conditions"
    trainer = Trainer(conf)
    trainer.train()
    trainer.close()


def val(conf):
    conf.__post_init__()
    conf.num_epochs = 1
    conf.__post_init__()
    trainer = Trainer(conf)
    trainer.validate()
    trainer.close()


def test(conf):
    conf.num_epochs = 1
    conf.__post_init__()

    trainer = Trainer(conf)
    trainer.test()
    trainer.close()

def setup_arg_parser():
    """
    Parse main arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Mode for running model (train, val, test)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Sub-path in DATA_DIR to checkpoint file (needed when val or test)")
    return parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    conf = get_chP96_clf_5cond_conf()
    args = setup_arg_parser()
    assert (args.mode not in ["val", "test"]) or args.checkpoint is not None, "Checkpoint must be given in val or test mode"

    if args.checkpoint is not None:
        conf.checkpoint["name"] = os.path.join(DATA_DIR, str(args.checkpoint))
    if args.mode == "train":
        train(conf)
    elif args.mode == "val":
        val(conf)
    elif args.mode == "test":
        test(conf)
    else:
        raise ValueError("Unknown mode")
