# Config for humanact12_22 conditional skeleton generation experiment

# --- General ---
exp_base_name = "humanact1222"
seed = 2
multi_gpu = "1"

# --- Dataloader ---
dataset_name = "humanact12_22"
motion_path = "./data/humanact12/new_joint_vecs_humanact"
old_new_map_path = "./data/humanact12/humanact12_old_new_map.json"
train_stat = "./data/humanact12/train_mean_std"
test_stat = "./data/humanact12/test_mean_std"
batch_size = 128
num_workers = 0

# --- Model ---
model = "mdm_condition"
input_feats = 75
njoints = 263
nfeats = 1
num_actions = 34

# --- Training ---
num_epoch = 1000
lr = 1e-4
lr_step_size = 200
step = 10
training = True

# --- Checkpoint ---
load_from = "humanact1222_0.0_latest_val.pth"

# --- Diffusion ---
renoise = 5  # effectively disabled
sub_remove = 0.
view_remove = 0
dropout_inf = 0

# --- Dataset ---
window_size = 48
unit_length = 4
bs_train = 128

# --- Augmentation ---
aug = True
# other_aug = True

# --- Sampling ---
num_frames = 48