"""Sensor-level augmentation for raw inertial windows.

Augmentation hanya diterapkan pada data pelatihan model 1D-CNN. Data validasi
dan data uji selalu bersih. Transformasi yang dipakai: jitter Gaussian,
scaling per channel, dan pergeseran waktu melingkar. Rotasi triaksial tidak
dipakai karena memerlukan penanganan gravitasi yang eksplisit.
"""

from __future__ import annotations

import numpy as np


def augment_signal_window(
    window: np.ndarray,
    rng: np.random.Generator,
    channel_scale: np.ndarray | None = None,
    jitter_sigma: float = 0.02,
    scale_range: tuple[float, float] = (0.9, 1.1),
    max_shift: int = 8,
) -> np.ndarray:
    """Augment one ``(window, channel)`` array using a seeded generator."""
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("window harus 2D (window, channel)")
    output = values.copy()
    n_channels = values.shape[1]

    if channel_scale is None:
        channel_scale = values.std(axis=0)
    channel_scale = np.asarray(channel_scale, dtype=np.float32).reshape(-1)
    if channel_scale.shape[0] != n_channels:
        raise ValueError("channel_scale tidak sesuai jumlah channel")
    channel_scale = np.where(channel_scale < 1e-8, 1.0, channel_scale)

    jitter = rng.normal(0.0, jitter_sigma, size=values.shape).astype(np.float32)
    output += jitter * channel_scale.reshape(1, -1)

    scales = rng.uniform(scale_range[0], scale_range[1], size=n_channels).astype(np.float32)
    output *= scales.reshape(1, -1)

    if max_shift > 0:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        output = np.roll(output, shift=shift, axis=0)
    return output


def augment_batch_numpy(
    windows: np.ndarray,
    rng: np.random.Generator,
    channel_scale: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    batch = np.asarray(windows)
    if batch.ndim != 3:
        raise ValueError("windows harus 3D (n, window, channel)")
    if channel_scale is None:
        channel_scale = batch.std(axis=(0, 1))
    return np.stack(
        [
            augment_signal_window(window, rng, channel_scale=channel_scale, **kwargs)
            for window in batch
        ]
    ).astype(np.float32)


def augment_batch_torch(batch, channel_std, rng: np.random.Generator):
    """Augment a torch tensor shaped ``(batch, channel, window)``.

    ``channel_std`` should be a NumPy array of shape ``(channel,)`` computed
    from the training data.
    """
    import torch

    if batch.ndim != 3:
        raise ValueError("batch harus 3D (batch, channel, window)")
    n_batch, n_channels, _ = batch.shape
    scale = rng.uniform(0.9, 1.1, size=(n_batch, n_channels, 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.02, size=tuple(batch.shape)).astype(np.float32)
    channel_scale = np.asarray(channel_std, dtype=np.float32).reshape(1, n_channels, 1)
    channel_scale = np.where(channel_scale < 1e-8, 1.0, channel_scale)

    output = batch * torch.from_numpy(scale) + torch.from_numpy(noise * channel_scale)
    shifts = rng.integers(-8, 9, size=n_batch)
    if np.any(shifts):
        shifted = torch.empty_like(output)
        for index, shift in enumerate(shifts):
            shifted[index] = torch.roll(output[index], shifts=int(shift), dims=1)
        output = shifted
    return output
