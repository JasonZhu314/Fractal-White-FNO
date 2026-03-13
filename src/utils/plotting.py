import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_losses(train_losses, test_losses, save_dir, model_name):
    """Plot training and test relative L2 loss curves.

    Args:
        train_losses: list of per-epoch training relative L2 losses
        test_losses: list of per-epoch test relative L2 losses
        save_dir: directory to save the figure
        model_name: name for the title and filename (e.g. "FNO" or "UFNO")
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, train_losses, label='Train Rel. L2', linewidth=1.5)
    ax.semilogy(epochs, test_losses, label='Test Rel. L2', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative L2 Loss')
    ax.set_title(f'{model_name} — 1D Burgers\' Equation')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(save_dir, f'{model_name}_loss_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curve saved to {path}')


def plot_predictions(model, x_test, y_test, grid, save_dir, model_name,
                     x_normalizer=None, y_normalizer=None, device=None,
                     sample_indices=None):
    """Plot model predictions vs ground truth for selected test samples.

    Produces one figure with rows of subplots. Each row shows one test sample:
        - Left panel:  initial condition a(x) and ground truth u(x,t=1)
        - Right panel: ground truth vs prediction, with pointwise error shaded

    Args:
        model: trained nn.Module
        x_test: (n_test, n_x, in_dim) tensor — raw (un-normalized) test inputs
        y_test: (n_test, n_x, 1) tensor — raw test targets
        grid: 1D numpy array of spatial coordinates
        save_dir: directory to save the figure
        model_name: name for the title and filename
        x_normalizer: UnitGaussianNormalizer fitted on training x (or None)
        y_normalizer: UnitGaussianNormalizer fitted on training y (or None)
        device: torch device
        sample_indices: list of test sample indices to plot (default: 4 evenly spaced)
    """
    import torch

    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device
    if sample_indices is None:
        n = x_test.shape[0]
        sample_indices = [0, n // 4, n // 2, 3 * n // 4]

    n_samples = len(sample_indices)

    # Run inference
    model.eval()
    x_input = x_test[sample_indices]  # (n_samples, n_x, in_dim)
    if x_normalizer is not None:
        x_input = x_normalizer.encode(x_input)

    with torch.no_grad():
        pred = model(x_input.to(device)).cpu()  # (n_samples, n_x, 1)

    if y_normalizer is not None:
        pred = y_normalizer.decode(pred)

    pred = pred.squeeze(-1).numpy()                       # (n_samples, n_x)
    truth = y_test[sample_indices].squeeze(-1).numpy()    # (n_samples, n_x)
    initial = x_test[sample_indices, :, 0].numpy()        # (n_samples, n_x) — a(x)

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        idx = sample_indices[i]
        error = np.abs(truth[i] - pred[i])
        rel_l2 = np.linalg.norm(truth[i] - pred[i]) / np.linalg.norm(truth[i])

        # Left: initial condition + ground truth
        ax = axes[i, 0]
        ax.plot(grid, initial[i], color='C2', linewidth=1.2, label='a(x) [t=0]')
        ax.plot(grid, truth[i], color='C0', linewidth=1.2, label='u(x) [t=1]')
        ax.set_title(f'Sample #{idx} — Input & Ground Truth')
        ax.set_xlabel('x')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: prediction vs truth + error
        ax = axes[i, 1]
        ax.plot(grid, truth[i], color='C0', linewidth=1.5, label='Ground Truth')
        ax.plot(grid, pred[i], color='C3', linewidth=1.5, linestyle='--', label='Prediction')
        ax.fill_between(grid, truth[i] - error, truth[i] + error,
                        alpha=0.15, color='C3', label='|Error|')
        ax.set_title(f'Prediction — Rel. L2 = {rel_l2:.4f}')
        ax.set_xlabel('x')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} — 1D Burgers\' Equation Predictions', fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(save_dir, f'{model_name}_predictions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Predictions plot saved to {path}')
