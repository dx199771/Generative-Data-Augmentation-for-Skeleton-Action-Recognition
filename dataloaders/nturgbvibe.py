import os
import random
import pickle as pkl

import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin

# ------------------------------------------------------------------ #
# Label mappings                                                       #
# ------------------------------------------------------------------ #

ACTION_DICT = {
    'A006': 'pickup',
    'A007': 'throw',
    'A008': 'sitting down',
    'A009': 'standing up (from sitting position)',
    'A022': 'cheer up',
    'A023': 'hand waving',
    'A024': 'kicking something',
    'A038': 'salute',
    'A080': 'squat down',
    'A093': 'shake fist',
    'A099': 'running on the spot',
    'A100': 'butt kicks (kick backward)',
    'A102': 'side kick',
}

LABEL_TO_IDX = {label: idx for idx, label in enumerate(ACTION_DICT.values())}
NUM_CLASSES  = len(LABEL_TO_IDX)  # 13


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def one_hot_encode(index: int, num_classes: int) -> np.ndarray:
    """Convert a class index to a one-hot vector."""
    vec = np.zeros(num_classes)
    vec[index] = 1
    return vec


def _get_label(file_path: str) -> np.ndarray:
    """Derive a one-hot label from the last 4 chars of the filename stem."""
    code  = os.path.basename(file_path).split(".")[0][-4:]
    label = ACTION_DICT[code]
    return one_hot_encode(LABEL_TO_IDX[label], NUM_CLASSES)


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #

class VQMotionDataset(data.Dataset):
    def __init__(self, cfg):
        self.window_size = cfg.window_size
        self.unit_length = cfg.unit_length
        self.lengths     = []

        for split in ("train", "test"):
            split_pkl = pjoin(cfg.motion_base_path, f"vibe_48_new_{split}.pkl")
            with open(split_pkl, "rb") as f:
                split_idx = pkl.load(f)

            data_list, labels, names = [], [], []

            for path in [os.path.join(cfg.motion_path, i) for i in split_idx]:
                motion = np.load(path)

                # Skip NaN or too-short sequences
                if np.isnan(motion).any() or motion.shape[0] < self.window_size:
                    continue

                labels.append(_get_label(path))
                names.append(os.path.basename(path).split(".")[0])
                data_list.append(motion)
                self.lengths.append(motion.shape[0] - self.window_size)

            if split == "train":
                # Optionally subsample training data
                if cfg.sub_remove > 0:
                    indices   = random.sample(
                        range(len(data_list)),
                        int(len(data_list) * (1 - cfg.sub_remove))
                    )
                    data_list = [data_list[i] for i in indices]
                    labels    = [labels[i]    for i in indices]
                    names     = [names[i]     for i in indices]

                self.data  = data_list
                self.label = labels
                self.name  = names
            else:
                self.test_data  = data_list
                self.test_label = labels
                self.test_name  = names

        # Load normalisation statistics
        self.mean      = np.load(pjoin(cfg.train_stat, "Mean.npy"))
        self.std       = np.load(pjoin(cfg.train_stat, "Std.npy"))
        self.test_mean = np.load(pjoin(cfg.test_stat,  "Mean.npy"))
        self.test_std  = np.load(pjoin(cfg.test_stat,  "Std.npy"))

        print(f"Total training motions: {len(self.data)}")

    # -------------------------------------------------------------- #

    def inv_transform(self, data):
        """Undo z-normalisation."""
        return data * self.std + self.mean

    def compute_sampling_prob(self) -> np.ndarray:
        """Length-proportional sampling weights."""
        prob = np.array(self.lengths, dtype=np.float32)
        return prob / prob.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        label  = self.label[item]

        # Random temporal crop
        idx    = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx : idx + self.window_size]

        # Z-normalisation
        motion = (motion - self.mean) / self.std

        return motion.astype(np.float32), label


# ------------------------------------------------------------------ #
# Loader factory                                                       #
# ------------------------------------------------------------------ #

def get_dataset_loader(cfg):
    dataset = VQMotionDataset(cfg)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg.bs_train,
        shuffle     = True,
        num_workers = cfg.num_workers,
        drop_last   = False,
    )
    return loader, dataset
