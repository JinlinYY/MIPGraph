from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from torch import nn


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def make_sklearn_model(kind: str = "extratrees", seed: int = 42):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    return ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=-1)


class FingerprintMLP(nn.Module):
    def __init__(self, in_dim: int = 2050, hidden_dim: int = 512, dropout: float = 0.15, out_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
