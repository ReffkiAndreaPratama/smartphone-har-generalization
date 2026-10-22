"""Model definitions: Random Forest, SVM pipeline and a compact 1D-CNN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from .preprocessing import RawSignalStandardizer
from .utils import ProjectError, RAW_CHANNELS


class ModelError(ProjectError):
    pass


SKLEARN_MODEL_NAMES = ("random_forest", "svm")
MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "cnn": "1D-CNN",
}


def make_sklearn_model(
    name: str, params: dict[str, Any] | None = None, seed: int = 42, n_jobs: int = -1
):
    params = dict(params or {})
    if name == "random_forest":
        allowed = {"n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight"}
        filtered = {key: value for key, value in params.items() if key in allowed}
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **filtered,
        )
    if name == "svm":
        svm_params = {
            "kernel": params.get("kernel", "rbf"),
            "C": params.get("C", 1.0),
            "gamma": params.get("gamma", "scale"),
            "cache_size": params.get("cache_size", 1000),
            "probability": bool(params.get("probability", False)),
            "random_state": seed,
        }
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(**svm_params)),
            ]
        )
    raise ModelError(f"Model sklearn tidak dikenal: {name}")


def model_param_grid(config: dict[str, Any], name: str) -> dict[str, list[Any]]:
    grid = config.get("models", {}).get(name, {}).get("selection_grid", {})
    if not grid:
        return {}
    return {key: list(values) for key, values in grid.items()}


def base_sklearn_params(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = dict(config.get("models", {}).get(name, {}))
    section.pop("selection_grid", None)
    section.pop("inner_folds", None)
    return section


@dataclass
class CNNConfig:
    """Training configuration for the 1D-CNN."""

    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (5, 5, 3)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 7
    val_subject_fraction: float = 0.2
    augment: bool = False
    augment_copies: int = 1
    seed: int = 42
    n_classes: int = 6
    in_channels: int = len(RAW_CHANNELS)
    window_size: int = 128

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        seed: int = 42,
        overrides: dict[str, Any] | None = None,
    ) -> "CNNConfig":
        section = dict(config.get("models", {}).get("cnn", {}))
        section["conv_channels"] = tuple(section.get("conv_channels", (32, 64, 128)))
        section["kernel_sizes"] = tuple(section.get("kernel_sizes", (5, 5, 3)))
        section["seed"] = seed
        if overrides:
            section.update(overrides)
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in section.items() if key in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conv_channels"] = list(self.conv_channels)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        return payload


class HARCNN(nn.Module):
    """Compact 1D-CNN for 128x9 inertial windows."""

    def __init__(
        self,
        in_channels: int = len(RAW_CHANNELS),
        n_classes: int = 6,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (5, 5, 3),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels dan kernel_sizes harus sama panjang")
        layers: list[nn.Module] = []
        current_channels = in_channels
        for block_index, (out_channels, kernel_size) in enumerate(
            zip(conv_channels, kernel_sizes)
        ):
            layers.extend(
                [
                    nn.Conv1d(current_channels, out_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if block_index < 2:
                layers.append(nn.MaxPool1d(kernel_size=2))
            current_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, window)
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


class TorchHARModel:
    """Serializable wrapper bundling the CNN, preprocessing and metadata."""

    def __init__(
        self,
        model: HARCNN,
        config: CNNConfig,
        normalizer: RawSignalStandardizer,
        class_names: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.class_names = list(class_names)
        self.metadata = dict(metadata or {})
        self.history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None

    def _prepare(self, X_raw: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        values = self.normalizer.transform(X_raw, subjects)
        # (n, window, channel) -> (n, channel, window)
        return np.transpose(values, (0, 2, 1)).astype(np.float32)

    def predict_proba(
        self,
        X_raw: np.ndarray,
        subjects: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        x = torch.from_numpy(self._prepare(X_raw, subjects))
        self.model.eval()
        outputs = []
        batch_size = int(batch_size or self.config.batch_size)
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                logits = self.model(x[start : start + batch_size])
                outputs.append(torch.softmax(logits, dim=1))
        return torch.cat(outputs).cpu().numpy()

    def predict(
        self,
        X_raw: np.ndarray,
        subjects: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(X_raw, subjects, batch_size=batch_size), axis=1)

    def save(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "class_names": self.class_names,
            "metadata": {**self.metadata, **(extra_metadata or {})},
            "history": self.history,
            "best_epoch": self.best_epoch,
            "architecture": "HARCNN",
            "n_parameters": count_parameters(self.model),
        }
        torch.save(payload, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TorchHARModel":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("architecture") != "HARCNN":
            raise ModelError(f"Arsitektur tidak dikenal pada {path}")
        config = CNNConfig(**payload["config"])
        model = HARCNN(
            in_channels=config.in_channels,
            n_classes=config.n_classes,
            conv_channels=tuple(config.conv_channels),
            kernel_sizes=tuple(config.kernel_sizes),
            dropout=config.dropout,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        normalizer = RawSignalStandardizer.from_dict(payload["normalizer"])
        instance = cls(
            model=model,
            config=config,
            normalizer=normalizer,
            class_names=payload["class_names"],
            metadata=payload.get("metadata", {}),
        )
        instance.history = payload.get("history", [])
        instance.best_epoch = payload.get("best_epoch")
        return instance


def predict_scores(model: Any, X: np.ndarray) -> tuple[np.ndarray, str]:
    """Return class scores and a label describing whether they are calibrated."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X), dtype=np.float64)
        return scores, "probability"
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=np.float64)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True), "normalized_decision_score"
    raise ModelError("Model tidak menyediakan predict_proba atau decision_function")


def model_display_name(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)
