"""Fractal-White-FNO: spectral whitening wrapper for neural operators.

Applies hardcoded spectral pre-emphasis (|k|^alpha gain) before the backbone
and de-emphasis (|k|^{-alpha}) after, based on Kolmogorov's k^{-5/3} energy
cascade law.  The whitening forces the backbone to give equal attention to
high-frequency features (e.g. shockwaves) that would otherwise be drowned out.

No learnable parameters are added — the whitening is a fixed signal-processing
transform.
"""

import torch
import torch.nn as nn


class SpectralFilter1d(nn.Module):
    """Applies a power-law gain |k|^alpha in the frequency domain.

    Args:
        n_x: spatial resolution (number of grid points).
        alpha: exponent for the gain curve.  Positive = amplify high freq,
               negative = attenuate high freq.
        max_gain: if set, clamp the gain tensor to this value (useful for
                  positive alpha to avoid noise explosion at high k).
    """

    def __init__(self, n_x: int, alpha: float = 5 / 3, max_gain: float | None = 1000.0):
        super().__init__()
        self.n_x = n_x

        # rfft of a real signal of length n_x has n_x//2 + 1 frequency bins
        n_freq = n_x // 2 + 1
        k = torch.arange(n_freq, dtype=torch.float32)

        # k=0 (DC) gets gain 1.0; all others get |k|^alpha
        gain = torch.ones(n_freq)
        gain[1:] = k[1:].pow(alpha)

        if max_gain is not None:
            gain = gain.clamp(max=max_gain)

        # shape: (n_freq,) — broadcastable with rfft output
        self.register_buffer("gain", gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral filtering along the last dimension.

        Args:
            x: (batch, n_x) or (batch, n_x, 1).

        Returns:
            Filtered tensor, same shape as input.
        """
        squeezed = False
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            squeezed = True

        x_ft = torch.fft.rfft(x, dim=-1)
        x_ft = x_ft * self.gain
        out = torch.fft.irfft(x_ft, n=self.n_x, dim=-1)

        if squeezed:
            out = out.unsqueeze(-1)
        return out


class FractalWhiteFNO1d(nn.Module):
    """Fractal-White wrapper around any 1D neural operator backbone.

    Pipeline:
        1. Whiten selected input channels via |k|^{alpha} pre-emphasis
        2. Run backbone (e.g. UFNO1d, FNO1d)
        3. De-emphasize all output channels via |k|^{-alpha}

    Args:
        backbone: any nn.Module with signature (batch, n_x, in_dim) -> (batch, n_x, out_dim).
        n_x: spatial resolution.
        alpha: spectral exponent (default 5/3 from Kolmogorov theory).
        max_gain: cap on pre-emphasis gain (default 1000.0).
        whiten_channels: list of input channel indices to whiten (default [0]).
    """

    def __init__(self, backbone: nn.Module, n_x: int, alpha: float = 5 / 3,
                 max_gain: float = 1000.0, whiten_channels: list[int] | None = None):
        super().__init__()
        self.backbone = backbone
        self.whiten_channels = whiten_channels if whiten_channels is not None else [0]

        # Pre-emphasis: amplify high frequencies
        self.whitener = SpectralFilter1d(n_x, alpha=alpha, max_gain=max_gain)
        # De-emphasis: restore original spectral shape (gains < 1, no cap needed)
        self.deemphasis = SpectralFilter1d(n_x, alpha=-alpha, max_gain=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_x, in_dim)

        Returns:
            (batch, n_x, out_dim)
        """
        # 1. Whiten selected input channels (no in-place ops for autograd)
        channels = []
        for ch in range(x.shape[-1]):
            col = x[:, :, ch]
            if ch in self.whiten_channels:
                col = self.whitener(col)
            channels.append(col)
        x = torch.stack(channels, dim=-1)

        # 2. Backbone
        out = self.backbone(x)  # (batch, n_x, out_dim)

        # 3. De-emphasize all output channels (no in-place ops for autograd)
        out_channels = []
        for ch in range(out.shape[-1]):
            out_channels.append(self.deemphasis(out[:, :, ch]))
        out = torch.stack(out_channels, dim=-1)

        return out

    def count_params(self):
        """Learnable param count — delegates to backbone (whitening adds zero)."""
        return self.backbone.count_params()
