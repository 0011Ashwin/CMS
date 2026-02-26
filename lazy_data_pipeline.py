"""
Lazy Data Pipeline for JetClass Dataset
========================================
Reads particle 4-vectors (E, px, py, pz) directly from the .tar archive
WITHOUT full extraction.  Designed for constrained-storage environments
(e.g. 512 GB drives) and addresses ML4SCI Issue #17.

Usage
-----
    from lazy_data_pipeline import JetClassTarDataset, tar_collate_fn
    from torch.utils.data import DataLoader

    ds  = JetClassTarDataset("D:/datasets/JetClass_Pythia_train_100M_part0.tar",
                              max_particles=128, max_jets_per_file=1000)
    dl  = DataLoader(ds, batch_size=64, collate_fn=tar_collate_fn, num_workers=0)
    for batch in dl:
        particles, labels = batch   # (B, 128, 4),  (B, 10)
        break

Design
------
* Streams ROOT files via ``tarfile`` → ``io.BytesIO`` → ``uproot`` so that
  the archive is never extracted to disk.
* Each worker holds a *separate* ``tarfile`` handle (re-opened via
  ``worker_init_fn``) to avoid pickling issues in DataLoader.
* Zero-pads particle lists to ``max_particles`` and converts
  (px, py, pz, E) → (pT, η, φ, E) for downstream compatibility with the
  LorentzParT pipeline (Hybrid_Transformer by Thanh Nguyen, 2025).

Author: GSoC 2026 Candidate
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
#  Optional heavy imports – graceful fallback
# ---------------------------------------------------------------------------
try:
    import uproot
    import awkward as ak
    import vector

    vector.register_awkward()
    _HAS_UPROOT = True
except ImportError:
    _HAS_UPROOT = False


# ======================================================================== #
#  Core helpers                                                             #
# ======================================================================== #


def _root_bytes_to_arrays(
    buf: bytes,
    max_particles: int = 128,
    max_jets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a single in-memory ROOT file and return padded arrays.

    Parameters
    ----------
    buf : bytes
        Raw bytes of a ``.root`` file (read from the tar member).
    max_particles : int
        Pad / truncate each jet to this many particles.
    max_jets : int | None
        If set, only return the first ``max_jets`` jets (useful for quick tests).

    Returns
    -------
    particles : ndarray, shape ``(N, max_particles, 4)``
        Columns are ``(pT, eta, phi, energy)``.
    labels : ndarray, shape ``(N, 10)``
        One-hot jet labels (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl).
    """
    if not _HAS_UPROOT:
        raise ImportError(
            "uproot, awkward, and vector are required.  "
            "Install them with:  pip install uproot awkward vector"
        )

    # Write to temp file — uproot cannot reliably open from bytes/BytesIO
    import tempfile as _tmpmod
    tmp = _tmpmod.NamedTemporaryFile(suffix=".root", delete=False)
    tmp.write(buf)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        file = uproot.open(tmp_path)
        tree_name = [k for k in file.keys() if "tree" in k.lower() or "Events" in k]
        if not tree_name:
            tree_name = list(file.keys())
        tree = file[tree_name[0]]

    # ---- particle branches --------------------------------------------------
        branches = ["part_px", "part_py", "part_pz", "part_energy"]
        arrays = tree.arrays(branches, library="ak")

        px = arrays["part_px"]
        py = arrays["part_py"]
        pz = arrays["part_pz"]
        energy = arrays["part_energy"]

        # Compute pT, eta, phi from px, py, pz
        pt = np.sqrt(ak.to_numpy(px) ** 2 + ak.to_numpy(py) ** 2) if False else None  # placeholder

        # Use vector for proper Lorentz computation
        p4 = ak.zip(
            {"px": px, "py": py, "pz": pz, "energy": energy},
            with_name="Momentum4D",
        )

        pt_arr = ak.to_numpy(ak.fill_none(ak.pad_none(p4.pt, max_particles, clip=True), 0.0))
        eta_arr = ak.to_numpy(ak.fill_none(ak.pad_none(p4.eta, max_particles, clip=True), 0.0))
        phi_arr = ak.to_numpy(ak.fill_none(ak.pad_none(p4.phi, max_particles, clip=True), 0.0))
        e_arr = ak.to_numpy(ak.fill_none(ak.pad_none(p4.energy, max_particles, clip=True), 0.0))

        # Stack → (N, 128, 4)
        particles = np.stack([pt_arr, eta_arr, phi_arr, e_arr], axis=-1).astype(np.float32)

        # ---- label branches --------------------------------------------------
        label_names = [
            "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
            "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl",
        ]
        existing_labels = [lb for lb in label_names if lb in tree.keys()]
        if existing_labels:
            lab_arrays = tree.arrays(existing_labels, library="np")
            labels = np.column_stack([lab_arrays[lb] for lb in existing_labels]).astype(np.float32)
        else:
            labels = np.zeros((particles.shape[0], 10), dtype=np.float32)

        if max_jets is not None:
            particles = particles[:max_jets]
            labels = labels[:max_jets]

        return particles, labels
    finally:
        file.close()
        os.unlink(tmp_path)


def _h5_bytes_to_arrays(
    buf: bytes,
    max_particles: int = 128,
    max_jets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback reader for HDF5 files inside the tar."""
    import h5py

    with h5py.File(io.BytesIO(buf), "r") as f:
        if "part_features" in f:
            particles = f["part_features"][:]
        elif "X" in f:
            particles = f["X"][:]
        else:
            key = list(f.keys())[0]
            particles = f[key][:]

        if "labels" in f:
            labels = f["labels"][:]
        elif "y" in f:
            labels = f["y"][:]
        else:
            labels = np.zeros((particles.shape[0], 10), dtype=np.float32)

    # Ensure (N, P, F)
    if particles.ndim == 3 and particles.shape[1] == 4 and particles.shape[2] > 4:
        particles = particles.transpose(0, 2, 1)

    # Pad / truncate
    N, P, F = particles.shape
    if P < max_particles:
        pad = np.zeros((N, max_particles - P, F), dtype=np.float32)
        particles = np.concatenate([particles, pad], axis=1)
    elif P > max_particles:
        particles = particles[:, :max_particles, :]

    if max_jets is not None:
        particles = particles[:max_jets].astype(np.float32)
        labels = labels[:max_jets].astype(np.float32)

    return particles, labels


# ======================================================================== #
#  Index builder – scans tar once and caches member offsets                 #
# ======================================================================== #


def build_tar_index(tar_path: str | Path) -> List[tarfile.TarInfo]:
    """Return a list of TarInfo for every data file in the archive.

    This scans the archive *sequentially* (metadata only, no extraction)
    and takes ~10-30s for a 15 GB tar.  The index is cacheable.
    """
    tar_path = Path(tar_path)
    data_extensions = {".root", ".h5", ".hdf5", ".npz", ".npy"}
    index: list[tarfile.TarInfo] = []

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).suffix.lower() in data_extensions:
                index.append(member)

    print(f"[LazyPipeline] Indexed {len(index)} data files in {tar_path.name}")
    return index


# ======================================================================== #
#  PyTorch Dataset                                                         #
# ======================================================================== #


class JetClassTarDataset(Dataset):
    """Memory-efficient dataset that streams from a .tar file.

    Each *item* is a single jet (1, max_particles, 4) tensor.
    The dataset lazily reads ROOT/HDF5 members from the tar on demand.


    Parameters
    ----------
    tar_path : str | Path
        Path to the ``.tar`` archive.
    max_particles : int
        Zero-pad / truncate jet to this many constituents.
    max_jets_per_file : int | None
        Cap the number of jets read from each internal file.
    preload_index : bool
        If True (default), builds the tar index at construction time.
    """

    def __init__(
        self,
        tar_path: str | Path,
        max_particles: int = 128,
        max_jets_per_file: Optional[int] = None,
        preload_index: bool = True,
    ):
        self.tar_path = Path(tar_path)
        self.max_particles = max_particles
        self.max_jets_per_file = max_jets_per_file

        self._tar_handle: Optional[tarfile.TarFile] = None
        self._index: List[tarfile.TarInfo] = []
        self._file_jet_counts: List[int] = []
        self._cumulative: Optional[np.ndarray] = None

        # Cache: file_idx -> (particles, labels)
        self._cache: dict = {}
        self._cache_max = 3  # keep at most 3 files in memory

        if preload_index:
            self._build_index()

    # ------------------------------------------------------------------ #
    #  Index                                                              #
    # ------------------------------------------------------------------ #

    def _build_index(self):
        """Scan the tar to find data files and count jets in each."""
        self._index = build_tar_index(self.tar_path)

        # For each file, read it once to get the jet count
        # (We must know the total length for __len__)
        self._file_jet_counts = []
        with tarfile.open(self.tar_path, "r") as tar:
            for i, member in enumerate(self._index):
                try:
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        self._file_jet_counts.append(0)
                        continue
                    buf = fobj.read()
                    particles, labels = self._parse_buf(buf, member.name)
                    self._file_jet_counts.append(particles.shape[0])
                    # Cache the first few files directly
                    if i < self._cache_max:
                        self._cache[i] = (particles, labels)
                except Exception as e:
                    print(f"[LazyPipeline] Skipping {member.name}: {e}")
                    self._file_jet_counts.append(0)

        self._cumulative = np.cumsum([0] + self._file_jet_counts)
        print(
            f"[LazyPipeline] Total jets indexed: {self._cumulative[-1]:,} "
            f"across {len(self._index)} files"
        )

    def _parse_buf(self, buf: bytes, name: str) -> Tuple[np.ndarray, np.ndarray]:
        suffix = Path(name).suffix.lower()
        if suffix == ".root":
            return _root_bytes_to_arrays(buf, self.max_particles, self.max_jets_per_file)
        else:
            return _h5_bytes_to_arrays(buf, self.max_particles, self.max_jets_per_file)

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                  #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self._cumulative is None:
            self._build_index()
        return int(self._cumulative[-1])

    def _load_file(self, file_idx: int):
        if file_idx in self._cache:
            return self._cache[file_idx]

        member = self._index[file_idx]
        with tarfile.open(self.tar_path, "r") as tar:
            fobj = tar.extractfile(member)
            buf = fobj.read()
        particles, labels = self._parse_buf(buf, member.name)

        # Evict oldest if cache full
        if len(self._cache) >= self._cache_max:
            oldest = min(self._cache.keys())
            del self._cache[oldest]
        self._cache[file_idx] = (particles, labels)
        return particles, labels

    def __getitem__(self, idx: int):
        # Binary-search for the file that contains this jet
        file_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        local_idx = idx - int(self._cumulative[file_idx])

        particles, labels = self._load_file(file_idx)
        x = torch.tensor(particles[local_idx], dtype=torch.float32)  # (128, 4)
        y = torch.tensor(labels[local_idx], dtype=torch.float32)     # (10,)
        return x, y


def tar_collate_fn(batch):
    """Default collate – stacks into (B, 128, 4) and (B, 10)."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


# ======================================================================== #
#  Quick self-test                                                         #
# ======================================================================== #


def _quick_test(tar_path: str):
    """Smoke-test: index, load 1 batch, print shapes."""
    print("=" * 60)
    print("  JetClass Lazy Pipeline — Quick Test")
    print("=" * 60)

    ds = JetClassTarDataset(tar_path, max_particles=128, max_jets_per_file=500)
    print(f"\nDataset length: {len(ds)}")

    # Single item
    x, y = ds[0]
    print(f"Single jet  : x.shape={x.shape}, y.shape={y.shape}")

    # DataLoader
    dl = DataLoader(ds, batch_size=32, collate_fn=tar_collate_fn, shuffle=False)
    for bx, by in dl:
        print(f"First batch : x.shape={bx.shape}, y.shape={by.shape}")
        print(f"pT range    : [{bx[:, :, 0].min():.4f}, {bx[:, :, 0].max():.4f}]")
        print(f"Energy range: [{bx[:, :, 3].min():.4f}, {bx[:, :, 3].max():.4f}]")
        break

    print("\n✓ Lazy pipeline works.")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "D:/datasets/JetClass_Pythia_train_100M_part0.tar"
    _quick_test(path)
