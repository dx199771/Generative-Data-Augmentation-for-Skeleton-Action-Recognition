import os
import sys
import pickle as pkl

import numpy as np
import torch

import models
from utils import parse_args, init_gpu, init_seed



# ------------------------------------------------------------------ #
# Augmentation helpers                                                 #
# ------------------------------------------------------------------ #

def add_gaussian_noise(skeleton, mean=0.02, std=0.05):
    """Add Gaussian noise to a skeleton sequence."""
    return skeleton + np.random.normal(mean, std, skeleton.shape)


def random_scaling(skeleton, scale_range=(0.9, 1.1)):
    """Randomly scale a skeleton sequence."""
    return skeleton * np.random.uniform(*scale_range)


# ------------------------------------------------------------------ #
# Motion preparation                                                   #
# ------------------------------------------------------------------ #

def prepare_humanact12(input_motions, label, cfg):
    """Reshape humanact12_22 batch and build model_kwargs."""
    N, C, M = input_motions.shape
    motion = input_motions.permute(0, 2, 1).reshape(N, M, 1, C).float().cuda()
    model_kwargs = {
        'source_motion': motion,
        'label':         label.float().cuda(),
    }
    cfg.num_frames = 64
    return N, motion, model_kwargs


def prepare_ntu(input_motions, label, cfg, NN=1):
    """Reshape NTU-format batch and build model_kwargs."""
    N, C, T, V, M = input_motions.shape
    motion = (
        input_motions.permute(0, 4, 2, 3, 1)
                     .contiguous()
                     .view(N * M, T, V, C)
                     .reshape(-1, T, V * C)
                     .permute(0, 2, 1)
                     .unsqueeze(2)
                     .float()
                     .cuda()
    )
    label_rep = label.repeat_interleave(2, dim=0).float().cuda()
    motion    = motion.repeat(NN, 1, 1, 1)
    label_rep = label_rep.repeat(NN, 1)
    model_kwargs = {
        'source_motion': motion,
        'label':         label_rep,
    }
    cfg.num_frames = 120
    return N, motion, label_rep, model_kwargs


# ------------------------------------------------------------------ #
# Main sampling loop                                                   #
# ------------------------------------------------------------------ #

def sample():
    cfg    = parse_args()
    device = init_gpu(cfg)
    init_seed(cfg)

    # Select dataloader
    if "vibe" in cfg.dataset_name:
        from dataloaders.nturgbd_vibe import get_dataset_loader
    elif "humanact12_22" in cfg.dataset_name:
        from dataloaders.humanact12 import get_dataset_loader

    dataloader, dataset = get_dataset_loader(cfg=cfg)
    model, diffusion    = models.get_model(cfg)

    # Load checkpoint
    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        model.load_state_dict(
            torch.load(cfg.load_from, map_location='cpu')["model_state_dict"]
        )
    model.eval()

    # Output data structure (pyskl format)
    out_data = {"split": {"xsub_train": [], "xsub_val": []}, "annotations": []}
    NN      = 1 # how many times we sample the data x5 or x1.
    counter = 0

    # ---------------------------------------------------------------- #
    # Training split                                                    #
    # ---------------------------------------------------------------- #
    for i, (input_motions, label) in enumerate(dataloader):
        print(f"Iter {i}/{len(dataloader)}")

        is_humanact = (cfg.dataset_name == "humanact12_22")

        if is_humanact:
            N, motion, model_kwargs = prepare_humanact12(input_motions, label, cfg)
        else:
            N, motion, label, model_kwargs = prepare_ntu(input_motions, label, cfg, NN)

        # --- Diffusion sampling ---
        if cfg.aug:
            sample_out = diffusion.p_sample_loop(
                model,
                (NN * N * 1, model.njoints, model.nfeats, cfg.num_frames),
                clip_denoised = False,
                model_kwargs  = model_kwargs,
                skip_timesteps = 0,
                init_image    = None,
                progress      = True,
                dump_steps    = None,
                noise         = None,
                const_noise   = False,
            )

            # Reshape to (N, M, T, V, C)
            sample_out = sample_out[:, :cfg.njoints // 2, :, :]
            sample_out = (
                sample_out
                .reshape(NN * N * 2, cfg.njoints // 6, 3, cfg.num_frames)
                .permute(0, 3, 1, 2)
                .reshape(NN * N, 2, cfg.num_frames, cfg.njoints // 6, 3)
            )

            for idx, motion_sample in enumerate(sample_out):
                label_save = label[idx * 2].tolist().index(1)

                # Zero out second person if absent in original
                if not (input_motions.repeat(NN, 1, 1, 1, 1) != 0).any(
                    dim=(2, 3, 4)
                )[idx][1]:
                    sample_out[idx][1] = 0

                motion_np = motion_sample.cpu().detach().numpy()

                # Renoise threshold filter
                ref = input_motions.repeat(NN, 1, 1, 1, 1)[idx][:, :, :cfg.njoints // 6, :].numpy()
                if abs(np.sum(motion_np - ref)) > cfg.renoise:
                    continue

                out_data["split"]["xsub_train"].append(str(counter) + "aug")
                out_data["annotations"].append({
                    'frame_dir':    str(counter) + "aug",
                    'label':        label_save,
                    'keypoint':     motion_np,
                    'total_frames': cfg.num_frames,
                })
                counter += 1

        # --- Save original motions ---
        for idx, motion_orig in enumerate(input_motions):
            label_save     = label[idx * 2].tolist().index(1)
            import pdb; pdb.set_trace()
            motion_np      = motion_orig[:, :, :cfg.njoints // 6, :].numpy()
            out_data["split"]["xsub_train"].append(str(counter))
            out_data["annotations"].append({
                'frame_dir':    str(counter),
                'label':        label_save,
                'keypoint':     motion_np,
                'total_frames': cfg.num_frames,
            })
            counter += 1

    # ---------------------------------------------------------------- #
    # Test split                                                        #
    # ---------------------------------------------------------------- #
    test_data = dataset.test_data

    for j, motion in enumerate(test_data):
        data_numpy      = np.array(motion)
        valid_frame_num = np.sum(motion.sum(0).sum(-1).sum(-1) != 0)

        if not is_vibe:
            data_numpy = valid_crop_resize(data_numpy, valid_frame_num, [0.5, 1], 120)

        C, T, V, M      = data_numpy.shape
        motion_save     = data_numpy.transpose(3, 1, 2, 0)
        label_save      = dataset.test_label[j].tolist().index(1)

        out_data["split"]["xsub_val"].append(str(j) + "test")
        out_data["annotations"].append({
            'frame_dir':    str(j) + "test",
            'label':        label_save,
            'keypoint':     motion_save,
            'total_frames': cfg.num_frames,
        })

    # ---------------------------------------------------------------- #
    # Save output pickle                                                #
    # ---------------------------------------------------------------- #
   
    drop_str = str(int(cfg.dropout_inf * 10)).zfill(2)
    out_path = (
        f'./output_skeleton'
        f'{cfg.dataset_name}_{cfg.sub_remove}_aug_{drop_str}_renoise{cfg.renoise}_{NN}_times.pkl'
    )

    with open(out_path, 'wb') as f:
        pkl.dump(out_data, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    sample()