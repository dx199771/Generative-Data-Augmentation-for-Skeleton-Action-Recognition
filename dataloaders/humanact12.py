import os
import json
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
    "A0101": "warm_up_wristankle",       "A0102": "warm_up_pectoral",
    "A0103": "warm_up_eblowback",        "A0104": "warm_up_bodylean_right_arm",
    "A0105": "warm_up_bodylean_left_arm","A0106": "warm_up_bow_right",
    "A0107": "warm_up_bow_left",         "A0201": "walk",
    "A0301": "run",                      "A0401": "jump_handsup",
    "A0402": "jump_vertical",            "A0501": "drink_bottle_righthand",
    "A0502": "drink_bottle_lefthand",    "A0503": "drink_cup_righthand",
    "A0504": "drink_cup_lefthand",       "A0505": "drink_both_hands",
    "A0601": "lift_dumbbell_with_right_hand",
    "A0602": "lift_dumbbell_with_left_hand",
    "A0603": "lift_dumbbells_with_both_hands",
    "A0604": "lift_dumbbell_over_head",
    "A0605": "lift_dumbbells_with_both_hands_and_bend_legs",
    "A0701": "sit",
    "A0801": "eat_finger_right",         "A0802": "eat_pie_hamburger",
    "A0803": "eat_with_left_hand",       "A0901": "turn_steering_wheel",
    "A1001": "take_out_phone_call_and_put_phone_back",
    "A1002": "call_with_left_hand",      "A1101": "boxing_left_right",
    "A1102": "boxing_left_upwards",      "A1103": "boxing_right_upwards",
    "A1104": "boxing_right_left",        "A1201": "throw_right_hand",
    "A1202": "throw_both_hands",
}

LABEL_TO_IDX = {label: idx for idx, label in enumerate(ACTION_DICT.values())}
NUM_CLASSES = len(LABEL_TO_IDX)  # 34


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def one_hot_encode(index: int, num_classes: int) -> np.ndarray:
    """Convert a class index to a one-hot vector."""
    vec = np.zeros(num_classes)
    vec[index] = 1
    return vec


def _get_label(file_name: str, old_new_map: dict) -> np.ndarray:
    """Derive a one-hot label from a motion filename using the old→new map."""
    key = file_name[1:] if "M" in file_name else file_name
    
    action_code = old_new_map[key]
    action_code = action_code.split(".")[0][-5:]
    label = ACTION_DICT[action_code]
    return one_hot_encode(LABEL_TO_IDX[label], NUM_CLASSES)


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #

class VQMotionDataset(data.Dataset):
    def __init__(self, cfg):
        self.window_size = cfg.window_size
        self.unit_length = cfg.unit_length

        # Load old→new filename map
        with open(cfg.old_new_map_path, "rb") as f:
            old_new_map = json.load(f)

        self.lengths = []

        for split in ("train", "test"):
            split_pkl = pjoin("./data/humanact12", f"48_{split}.pkl")
            with open(split_pkl, "rb") as f:
                split_idx = pkl.load(f)

            data_list, labels, names = [], [], []

            for path in [os.path.join(cfg.motion_path, i) for i in split_idx]:
                motion = np.load(path)
                if motion.shape[0] < self.window_size:
                    continue

                file_name = os.path.basename(path)
                labels.append(_get_label(file_name, old_new_map))
                names.append(file_name.split(".")[0])
                data_list.append(motion)
                self.lengths.append(motion.shape[0] - self.window_size)

            if split == "train":
                # Optionally subsample training data
                if cfg.sub_remove > 0:
                    indices = random.sample(
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
        self.mean = np.load(pjoin(cfg.train_stat, "Mean.npy"))
        self.std  = np.load(pjoin(cfg.train_stat, "Std.npy"))
        self.test_mean = np.load(pjoin(cfg.test_stat, "Mean.npy"))
        self.test_std  = np.load(pjoin(cfg.test_stat, "Std.npy"))

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

        return motion.astype(np.float32), label.astype(np.float32)


# ------------------------------------------------------------------ #
# Loader                                                             #
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