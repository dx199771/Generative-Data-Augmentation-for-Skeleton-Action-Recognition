# Config for humanact12_22 conditional skeleton generation experiment

# --- General ---
exp_base_name = "nturgbd60_vibe"
seed = 2
multi_gpu = "1"

# --- Dataloader ---
dataset_name = "nturgbd60_sub_vibe"
motion_base_path = "./data/ntuvibe"
motion_path = "./data/ntuvibe/new_joint_vecs_vibe"
train_stat = "./data/ntuvibe/vibe_48_new_train"
test_stat = "./data/ntuvibe/vibe_48_new_test"
batch_size = 128
num_workers = 0

# --- Model ---
model = "mdm_condition"
input_feats = 75
njoints = 263
nfeats = 1
num_actions = 13

# --- Training ---
num_epoch = 1000
lr = 1e-4
lr_step_size = 200
step = 10
trainig = True

# --- Checkpoint ---
load_from = ""

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
# aug = False
# other_aug = True