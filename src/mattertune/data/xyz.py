from __future__ import annotations
import torch

import logging
from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.io import read
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import override
import copy

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class XYZDatasetConfig(DatasetConfigBase):
    type: Literal["xyz"] = "xyz"
    """Discriminator for the XYZ dataset."""

    src: str | Path
    """The path to the XYZ dataset."""
    
    down_sample: int | None = None
    """Down sample the dataset"""
    
    down_sample_refill: bool = False
    """Refill the dataset after down sampling to achieve the same length as the original dataset"""

    T: int = 1
    """Diffusion total time steps"""

    sigma_min: float 
    """Diffusion sigma min"""

    sigma_max: float 
    """Diffusion sigma max"""

    diffusion_type: str = "discrete"
    """Either discrete (eps prediction) or continous (score-matching)"""

    @override
    def create_dataset(self):
        return XYZDataset(self)


class XYZDataset(Dataset[ase.Atoms]):
    def __init__(self, config: XYZDatasetConfig):
        super().__init__()
        self.config = config

        atoms_list = read(str(self.config.src), index=":")
        assert isinstance(atoms_list, list), "Expected a list of Atoms objects"
        if self.config.down_sample is not None:
            ori_length = len(atoms_list)
            down_indices = np.random.choice(ori_length, self.config.down_sample, replace=False)
            if self.config.down_sample_refill:
                refilled_down_indices = []
                for _ in range((ori_length // self.config.down_sample)):
                    refilled_down_indices.extend(copy.deepcopy(down_indices))
                if len(refilled_down_indices) != ori_length:
                    res = np.random.choice(len(down_indices), ori_length - len(refilled_down_indices), replace=False)
                    refilled_down_indices.extend([down_indices[i] for i in res])
                new_atoms_list = [copy.deepcopy(atoms_list[i]) for i in refilled_down_indices]
                atoms_list = new_atoms_list
            else:
                new_atoms_list = [copy.deepcopy(atoms_list[i]) for i in down_indices]
                atoms_list = new_atoms_list
        self.atoms_list: list[Atoms] = atoms_list
        log.info(f"Loaded {len(self.atoms_list)} atoms from {self.config.src}")
    
    def vp_xt(self, x0, t, eps):
        beta_min = self.config.sigma_min
        beta_max = self.config.sigma_max
        t = torch.as_tensor(t, dtype=x0.dtype, device=x0.device)

        int_beta = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
        alpha = torch.exp(-0.5 * int_beta)
        sigma = torch.sqrt(1.0 - torch.exp(-int_beta))
        return x0 * alpha + eps * sigma

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        atoms = self.atoms_list[idx].copy()
        if self.config.diffusion_type == "discrete":
            T = self.config.T
            t = torch.randint(0, T, (1,)).item()
            sigma_min = self.config.sigma_min
            sigma_max = self.config.sigma_max
            sigma_t = sigma_min * (sigma_max / sigma_min) ** (t / (T))

            x_0 = torch.tensor(atoms.positions, dtype=torch.float32)
            eps = torch.randn_like(x_0)
            x_t = x_0 + eps * sigma_t

            atoms.positions = x_t.numpy()
            atoms.info['noise'] = (eps).numpy()
            atoms.info['t'] = t
            atoms.info['T'] = T
            atoms.info['type'] = 'discrete'
        elif self.config.diffusion_type == "vp":
            # sample t unifromly from 0, 1
            t = torch.rand(1).item()
            x0 = torch.tensor(atoms.positions, dtype=torch.float32)
            eps = torch.randn_like(x0)
            # translatoin drift stuff
            eps = eps - eps.mean(dim=0, keepdim=True)
            xt = self.vp_xt(x0, t, eps)
            atoms.positions = xt.numpy()
            atoms.info['noise'] = (eps).numpy()
            atoms.info['t'] = t
            atoms.info['type'] = 'vp'
        return atoms

    def __len__(self) -> int:
        return len(self.atoms_list)
