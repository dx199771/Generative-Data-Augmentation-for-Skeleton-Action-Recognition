import os
import torch
import random
import numpy as np
import argparse


def init_seed(args):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_gpu(cfg):
    """Set visible GPUs and return the compute device."""
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.multi_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def parse_args():
    """Parse command-line arguments and merge them into the mmcv config."""
    parser = argparse.ArgumentParser(description='Training config parser')

    parser.add_argument('--config', default='configs/cfg_humanact.py',
                        help='Path to the train config file')
    parser.add_argument('--dropout_inf', type=float, default=-1,
                        help='Dropout rate for inference (overrides config if set)')
    parser.add_argument('--load_from', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--sub_remove', type=float, default=-1,
                        help='Subject removal ratio (overrides config if set)')
    parser.add_argument('--renoise', type=int, default=-1,
                        help='Renoise step count (overrides config if set)')

    args = parser.parse_args()

    # Load base config from file, then override with any CLI arguments
    from mmcv import Config
    cfg = Config.fromfile(args.config)

    if args.renoise != -1:
        cfg.renoise = args.renoise
    if args.sub_remove != -1:
        cfg.sub_remove = args.sub_remove
    if args.load_from != '':
        cfg.load_from = args.load_from
    if args.dropout_inf != -1:
        cfg.dropout_inf = args.dropout_inf

    return cfg