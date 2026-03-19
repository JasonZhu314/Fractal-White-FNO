"""Train Fractal-White-FNO (1D) on 1D Burgers' equation.

Wraps a UFNO1d backbone with spectral whitening (|k|^{5/3} pre-emphasis)
and de-emphasis, forcing the network to give equal attention to high-frequency
shockwave features.

Dataset: burgers_data_R10.mat from the original FNO paper (Li et al., 2020).
    Download from: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
    Place at: data/burgers/burgers_data_R10.mat

Usage:
    python scripts/train_fw_fno_burgers.py
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Avoid OpenMP DLL conflict (numpy MKL vs torch)
import torch  # must be imported before numpy on this platform
import numpy as np
from scipy.io import loadmat

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.baselines.fno import FNO_train
from src.models.ufno1d import UFNO1d
from src.models.fractal_white_fno import FractalWhiteFNO1d
from src.utils.normalizer import UnitGaussianNormalizer
from src.utils.plotting import plot_losses, plot_predictions

torch.manual_seed(0)
np.random.seed(0)

###################################
# Configuration
# Same resolution/data as FNO and UFNO scripts for fair comparison
###################################
downsample_ratio = 16   # 8192 / 16 = 512 grid points
n_train = 1000
n_test = 200
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###################################
# Load data
###################################
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'burgers', 'burgers_data_R10.mat')
data_path = os.path.abspath(data_path)

if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    print("Please download burgers_data_R10.mat from:")
    print("  https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-")
    print(f"and place it at: data/burgers/burgers_data_R10.mat")
    sys.exit(1)

data = loadmat(data_path)
ndata, nnodes_ref = data["a"].shape
grid = np.linspace(0, 1, nnodes_ref)

# Downsample
features = np.stack((data["a"], data["u"]), axis=2)[:, ::downsample_ratio, :]
grid = grid[::downsample_ratio]
nnodes = nnodes_ref // downsample_ratio

# x_train, y_train are [n_data, n_x, n_channel] arrays
x_train = torch.from_numpy(
    np.stack((features[0:n_train, :, 0], np.tile(grid, (n_train, 1))), axis=-1).astype(np.float32)
)
y_train = torch.from_numpy(features[0:n_train, :, [1]].astype(np.float32))

x_test = torch.from_numpy(
    np.stack((features[-n_test:, :, 0], np.tile(grid, (n_test, 1))), axis=-1).astype(np.float32)
)
y_test = torch.from_numpy(features[-n_test:, :, [1]].astype(np.float32))

print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape:  {x_test.shape}")
print(f"y_test.shape:  {y_test.shape}")
print(f"Device: {device}")

###################################
# Construct model
# UFNO1d backbone with same hyperparams as baseline, wrapped in FractalWhiteFNO1d
###################################
k_max = 16
n_fno_layers = 3
n_ufno_layers = 3
n_total = n_fno_layers + n_ufno_layers

backbone = UFNO1d(
    modes=[k_max] * n_total,
    width=64,
    n_fno_layers=n_fno_layers,
    n_ufno_layers=n_ufno_layers,
    fc_dim=128,
    in_dim=2,
    out_dim=1,
    act='gelu',
    pad_ratio=0,
    unet_kernel_size=3,
    dropout_rate=0,
)

model = FractalWhiteFNO1d(
    backbone=backbone,
    n_x=nnodes,        # 512
    alpha=5 / 3,
    max_gain=1000.0,
    whiten_channels=[0],  # only whiten a(x), not grid coordinates
).to(device)

print(f"FractalWhiteFNO1d parameters: {model.count_params():,}")

###################################
# Output directory
###################################
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'fw_fno_burgers')
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

###################################
# Train
###################################
config = {
    "train": {
        "base_lr": 0.001,
        "weight_decay": 1.0e-4,
        "epochs": 500,
        "scheduler": "OneCycleLR",
        "batch_size": 20,
        "normalization_x": True,
        "normalization_y": True,
        "normalization_dim_x": [],
        "normalization_dim_y": [],
        "non_normalized_dim_x": 0,
        "non_normalized_dim_y": 0,
    }
}

train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = FNO_train(
    x_train, y_train, x_test, y_test, config, model,
    save_model_name=os.path.join(output_dir, "FW_FNO_burgers_model")
)

print(f"\nFinal Rel. Test L2 Loss: {test_rel_l2_losses[-1]:.6f}")

###################################
# Plot results
###################################
plot_losses(train_rel_l2_losses, test_rel_l2_losses, output_dir, 'FW-FNO')

# Recreate normalizers for inference (same config as FNO_train)
x_normalizer = UnitGaussianNormalizer(x_train, normalization_dim=[], non_normalized_dim=0)
y_normalizer = UnitGaussianNormalizer(y_train, normalization_dim=[], non_normalized_dim=0)
x_normalizer.to(device)
y_normalizer.to(device)

plot_predictions(model, x_test, y_test, grid, output_dir, 'FW-FNO',
                 x_normalizer=x_normalizer, y_normalizer=y_normalizer, device=device)
