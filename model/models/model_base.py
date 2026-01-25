import json
import torch
import hashlib
from abc import ABC, abstractmethod


class BaseTimeSeriesModel(torch.nn.Module, ABC):
    """
    Base class for all time-series models.
    Enforces:
      - model identity (type + version)
      - meta export / import
      - checkpoint save / load
    """

    MODEL_TYPE: str = "base"
    MODEL_VERSION: int = 0

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls) -> dict:
        return {
            "model_type": cls.MODEL_TYPE,
            "model_version": cls.MODEL_VERSION,
        }

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    @abstractmethod
    def export_meta(self) -> dict:
        """
        Return model-specific meta needed to reconstruct architecture.
        Must NOT include dataset-dependent info (window, feature_cols).
        """
        pass

    @classmethod
    @abstractmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + state_dict.
        """
        pass

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        model_path: str,
        meta_path: str,
        **extra_meta,
    ):
        """
        Save model weights + meta.
        extra_meta: dataset / training related info
                    (window, feature_cols, label_col, classes, etc.)
        """
        # 1) save weights
        torch.save(
            {
                "state_dict": self.state_dict(),
                "architecture": self.identity(),
            },
            model_path,
        )

        # 2) save meta
        meta = {
            **self.identity(),
            **self.export_meta(),
            **extra_meta,
        }

        meta["arch_hash"] = self.architecture_hash(meta)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    @classmethod
    def load_checkpoint(cls, model_path: str, meta_path: str, device):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        state = torch.load(model_path, map_location=device)

        # identity check
        if meta["model_type"] != cls.MODEL_TYPE:
            raise ValueError(f"Model type mismatch: {meta['model_type']} != {cls.MODEL_TYPE}")
        if meta["model_version"] != cls.MODEL_VERSION:
            raise ValueError(
                f"Model version mismatch: {meta['model_version']} != {cls.MODEL_VERSION}"
            )

        # hash check
        expected = meta.get("arch_hash")
        actual = cls.architecture_hash(meta)
        if expected and expected != actual:
            raise RuntimeError("Architecture hash mismatch! Model definition changed.")

        model = cls.build_from_meta(meta, state, device)
        model.eval()
        return model, meta

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    @staticmethod
    def architecture_hash(meta: dict) -> str:
        """
        Hash architecture-related fields only.
        Prevents silent mismatch between training & inference.
        """
        keys = sorted(k for k in meta.keys() if k not in {
            "window", "feature_cols", "label_col", "classes", "arch_hash"
        })
        payload = json.dumps({k: meta[k] for k in keys}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()
