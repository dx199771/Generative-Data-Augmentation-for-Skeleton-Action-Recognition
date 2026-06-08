from tqdm import tqdm

import models
from utils import *
from models.sampler import UniformSampler

import torch.optim as optim
from torch.optim import lr_scheduler


def train():
    # Parse config and initialize device/seed
    cfg = parse_args()
    device = init_gpu(cfg)
    init_seed(cfg)

    # Select dataloader based on dataset name
    if "nturgbvibe" in cfg.dataset_name:
        from dataloaders.nturgbvibe import get_dataset_loader
    elif "humanact12_22" in cfg.dataset_name:
        from dataloaders.humanact12 import get_dataset_loader

    print(cfg)
    dataloader, dataset = get_dataset_loader(cfg=cfg)

    # Build model and diffusion process
    model, diffusion = models.get_model(cfg)

    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=0.1)

    # Optionally resume from a checkpoint
    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    model.train()
    best_epoch = 0
    best_loss = float('inf')

    for epoch in range(cfg.num_epoch):
        all_loss_dict = {}

        for motion, label in tqdm(dataloader):
            # Reshape motion: (N, C, M) -> (N, M, 1, C) and move to GPU
            N, C, M = motion.shape
            motion = motion.permute(0, 2, 1).reshape(N, M, 1, C).cuda()
            label = label.cuda()

            bs, feats, _, frame = motion.shape

            # Sample diffusion timesteps uniformly
            t, weights = UniformSampler(diffusion).sample(bs, 'cuda')

            # Compute diffusion training losses
            all_loss, model_output = diffusion.training_losses(model, motion, t, label=label)
            loss = all_loss["loss"]

            # Accumulate losses for logging
            if not all_loss_dict:
                all_loss_dict = {k: [v.item()] for k, v in all_loss.items()}
            else:
                for i, k in enumerate(all_loss_dict):
                    all_loss_dict[k].append(list(all_loss.values())[i].item())

            # Backprop and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Track best epoch by average epoch loss
        avg_loss = sum(all_loss_dict["loss"]) / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch

        print(
            f"Epoch {epoch}  "
            f"Loss: { {k: sum(v) / len(dataloader) for k, v in all_loss_dict.items()} }  "
            f"Best epoch: {best_epoch}  Best loss: {best_loss:.4f}"
        )

        # Save checkpoint when current batch loss improves
        if loss < best_loss:
            save_dir = f'ckpts/{cfg.exp_base_name}/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                },
                f'{save_dir}{cfg.exp_base_name}_{cfg.sub_remove}_latest_val.pth'
            )


if __name__ == "__main__":
    train()