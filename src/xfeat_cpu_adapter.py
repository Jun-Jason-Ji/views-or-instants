"""Explicit-device adapter for the pinned, unmodified official XFeat inference code.

The only source transformation redirects two imports into a private namespace.
The official auto-CUDA constructor is never called. CPU is the default, and an
explicit CUDA device can be supplied by a separately authorized GPU runner.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/xfeat_2026-09-13"
COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
WEIGHT_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
_EXPECTED = {
    "modules/xfeat.py": "385ccd31d095b0d4176b04e982088b85321b11ade4324f83b097ee6524f2a6e7",
    "modules/model.py": "d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964",
    "modules/interpolator.py": "d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da",
    "weights/xfeat.pt": WEIGHT_SHA256,
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "README.md": "9f05c442c4ef0ced48deefb51307f65e8d14f72b14fb3fd65e8f75bfd3e321a5",
}
_NAMESPACE = "_paper3_official_xfeat_e92685f"


def verify_vendor(root: Path = VENDOR) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("Unexpected XFeat commit")
    entries = {row["path"]: row for row in manifest["files"]}
    if set(entries) != set(_EXPECTED):
        raise ValueError("Unexpected vendor file set")
    for rel, expected in _EXPECTED.items():
        data = (root / rel).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected or entries[rel]["sha256"] != expected:
            raise ValueError(f"Vendor hash mismatch: {rel}")
        if len(data) != entries[rel]["bytes"]:
            raise ValueError(f"Vendor size mismatch: {rel}")
    return {"commit": COMMIT, "checkpoint_sha256": WEIGHT_SHA256, "verified_files": len(_EXPECTED)}


def _load_official():
    verify_vendor()
    cached = sys.modules.get(_NAMESPACE + ".xfeat")
    if cached is not None:
        return cached
    package = types.ModuleType(_NAMESPACE)
    package.__path__ = [str(VENDOR / "modules")]
    sys.modules[_NAMESPACE] = package
    for name in ("model", "interpolator"):
        spec = importlib.util.spec_from_file_location(_NAMESPACE + "." + name, VENDOR / "modules" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    source = (VENDOR / "modules/xfeat.py").read_text(encoding="utf-8")
    redirects = {
        "from modules.model import *": f"from {_NAMESPACE}.model import *",
        "from modules.interpolator import InterpolateSparse2d": f"from {_NAMESPACE}.interpolator import InterpolateSparse2d",
    }
    for old, new in redirects.items():
        if source.count(old) != 1:
            raise ValueError("Official import layout changed")
        source = source.replace(old, new)
    module = types.ModuleType(_NAMESPACE + ".xfeat")
    module.__file__ = str(VENDOR / "modules/xfeat.py")
    module.__package__ = _NAMESPACE
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


_official = _load_official()


class XFeatAdapter(_official.XFeat):
    """Sparse XFeat features + mutual NN; no downloads or automatic device choice.

    `extract(image_uint8)` accepts HxW gray or HxWx3 RGB and returns NumPy
    keypoints Nx2, scores N, descriptors Nx64 in original pixel coordinates.
    `match_features(a, b)` returns indices Mx2 into those feature arrays.
    The threshold -1 follows official match_xfeat's public default.
    """

    def __init__(self, *, device: str = "cpu", top_k: int = 2000,
                 detection_threshold: float = 0.05, min_cossim: float = -1.0):
        torch.nn.Module.__init__(self)
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("device must explicitly be cpu or cuda[:index]")
        if top_k < 1 or not 0 <= detection_threshold <= 1 or not -1 <= min_cossim <= 1:
            raise ValueError("Invalid extraction or matching parameters")
        self.provenance = verify_vendor()
        self.dev = torch.device(device)
        # Construction and checkpoint decoding always begin on CPU.
        self.net = _official.XFeatModel()
        state = torch.load(VENDOR / "weights/xfeat.pt", map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.requires_grad_(False).eval().to(self.dev)
        self.top_k = int(top_k)
        self.detection_threshold = float(detection_threshold)
        self.min_cossim = float(min_cossim)
        self.interpolator = _official.InterpolateSparse2d("bicubic")
        self.kornia_available = False
        self.lighterglue = None
        self.eval()

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> dict[str, np.ndarray]:
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("Expected uint8 HxW gray or HxWx3 RGB")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("Expected exactly three RGB channels")
        if min(image.shape[:2]) < 32:
            raise ValueError("Each image dimension must be at least 32")
        result = self.detectAndCompute(np.ascontiguousarray(image), top_k=self.top_k,
                                       detection_threshold=self.detection_threshold)[0]
        return {key: value.detach().cpu().numpy().copy() for key, value in result.items()}

    @torch.inference_mode()
    def match_features(self, first: dict, second: dict) -> np.ndarray:
        descriptors = []
        for features in (first, second):
            desc = np.asarray(features["descriptors"])
            if desc.ndim != 2 or desc.shape[1] != 64 or not np.isfinite(desc).all():
                raise ValueError("Expected finite Nx64 descriptors")
            descriptors.append(torch.as_tensor(np.ascontiguousarray(desc), dtype=torch.float32, device=self.dev))
        if not len(descriptors[0]) or not len(descriptors[1]):
            return np.empty((0, 2), dtype=np.int64)
        idx0, idx1 = self.match(*descriptors, min_cossim=self.min_cossim)
        return torch.stack((idx0, idx1), dim=1).cpu().numpy().copy()

    def match_lighterglue(self, *args, **kwargs):
        raise RuntimeError("This frozen adapter only supports sparse XFeat mutual NN")
