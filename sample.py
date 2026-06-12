import os
import sys
import random
import pickle as pkl

import numpy as np
import torch
import matplotlib.pyplot as plt

import models
from utils import *

# git clone https://github.com/GuyTevet/motion-diffusion-model.git
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# ------------------------------------------------------------------ #
# Main sampling loop                                                   #
# ------------------------------------------------------------------ #

def sample():
    cfg    = parse_args()
    device = init_gpu(cfg)
    init_seed(cfg)

    # Select dataloader
    if "nturgbvibe" in cfg.dataset_name:
        from dataloaders.nturgbvibe import get_dataset_loader
    elif "humanact12_22" in cfg.dataset_name:
        from dataloaders.humanact12 import get_dataset_loader

    dataloader, dataset = get_dataset_loader(cfg=cfg)
    model, diffusion    = models.get_model(cfg)
    print(cfg)

    # Load checkpoint
    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        model.load_state_dict(
            torch.load(cfg.load_from, map_location='cpu')["model_state_dict"]
        )
    model.eval()

    # How many times the data needs to be generated
    NN = 5

    out_data  = {"split": {"xsub_train": [], "xsub_val": []}, "annotations": []}
    counter   = 0
    test_dist = []

    # ---------------------------------------------------------------- #
    # Training split                                                    #
    # ---------------------------------------------------------------- #
    for i, (input_motions_raw, label_raw) in enumerate(dataloader):
        print(f"Iter {i}/{len(dataloader)}")

        # Repeat batch NN times
        input_motions = input_motions_raw.repeat(NN, 1, 1)
        N, C, M       = input_motions.shape
        label         = label_raw.cuda().repeat(NN, 1)

        motion = input_motions.permute(0, 2, 1).reshape(N, M, 1, C).float().cuda()
        model_kwargs = {
            'source_motion': motion,
            'label':         label,
        }

        # --- Diffusion augmentation ---
        if cfg.aug:
            sample_out = diffusion.p_sample_loop(
                model,
                (N, model.njoints, model.nfeats, cfg.num_frames),
                clip_denoised  = False,
                model_kwargs   = model_kwargs,
                skip_timesteps = 0,
                init_image     = None,
                progress       = True,
                dump_steps     = None,
                noise          = None,
                const_noise    = False,
            )

            # Denormalise and recover 3D joint positions
            pred_xyz    = recover_from_ric(
                sample_out[:, :, 0, :].permute(0, 2, 1).detach().cpu().float()
                * dataset.std + dataset.mean, 22
            )
            pred_xyz_gt = recover_from_ric(
                input_motions.detach().cpu().float() * dataset.std + dataset.mean, 22
            )

            for idx in range(len(sample_out)):
                label_save = label[idx].tolist().index(1)
                dist_val   = abs(np.sum(
                    pred_xyz[idx].numpy() - pred_xyz_gt[idx].numpy()
                ))
                test_dist.append(dist_val)

                if dist_val > cfg.renoise:
                    continue

                out_data["split"]["xsub_train"].append(str(counter) + "aug")
                out_data["annotations"].append({
                    'frame_dir':    str(counter) + "aug",
                    'label':        label_save,
                    'keypoint':     pred_xyz[idx].unsqueeze(0).numpy(),
                    'total_frames': cfg.num_frames,
                })
                counter += 1

        # --- Other augmentation (noise/rotation) ---
        if cfg.other_aug:
            pred_xyz_gt_all = recover_from_ric(
                input_motions_raw.detach().cpu().float() * dataset.std + dataset.mean, 22
            )
            for idx in range(len(input_motions_raw)):
                label_save      = label_raw[idx].tolist().index(1)
                motion_aug      = add_gaussian_noise(pred_xyz_gt_all[idx].unsqueeze(0).numpy())

                out_data["split"]["xsub_train"].append(str(counter) + "aug")
                out_data["annotations"].append({
                    'frame_dir':    str(counter) + "aug",
                    'label':        label_save,
                    'keypoint':     motion_aug,
                    'total_frames': cfg.num_frames,
                })
                counter += 1

        # --- Save original motions ---
        pred_xyz_gt_raw = recover_from_ric(
            input_motions_raw.detach().cpu().float() * dataset.std + dataset.mean, 22
        )
        for idx in range(len(input_motions_raw)):
            label_save = label_raw[idx].tolist().index(1)
            out_data["split"]["xsub_train"].append(str(counter))
            out_data["annotations"].append({
                'frame_dir':    str(counter),
                'label':        label_save,
                'keypoint':     pred_xyz_gt_raw[idx].unsqueeze(0).numpy(),
                'total_frames': cfg.num_frames,
            })
            counter += 1

    # ---------------------------------------------------------------- #
    # Test split                                                        #
    # ---------------------------------------------------------------- #
    for j, motion in enumerate(dataset.test_data):
        idx          = random.randint(0, len(motion) - cfg.num_frames)
        motion_crop  = motion[idx : idx + cfg.num_frames]
        label_save   = dataset.test_label[j].tolist().index(1)
        pred_xyz_gt  = recover_from_ric(torch.tensor(motion_crop), 22).unsqueeze(0).numpy()

        out_data["split"]["xsub_val"].append(str(j) + "test")
        out_data["annotations"].append({
            'frame_dir':    str(j) + "test",
            'label':        label_save,
            'keypoint':     pred_xyz_gt,
            'total_frames': cfg.num_frames,
        })

    # ---------------------------------------------------------------- #
    # Save output pickle                                                #
    # ---------------------------------------------------------------- #
    sub_pct  = str(int(cfg.sub_remove * 100))
    drop_str = str(int(cfg.dropout_inf * 10)).zfill(2)

    if not cfg.aug and not cfg.other_aug:
        out_path = f'./output/{cfg.dataset_name}_{sub_pct}.pkl'
    elif cfg.other_aug:
        out_path = f'./output/{cfg.dataset_name}_{sub_pct}_rotation.pkl'
    else:
        out_path = (
            f'./output/{cfg.dataset_name}_{sub_pct}'
            f'_aug_{drop_str}_renoise{cfg.renoise}_{NN}times.pkl'
        )

    with open(out_path, 'wb') as f:
        pkl.dump(out_data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    sample()