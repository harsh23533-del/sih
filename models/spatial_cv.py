"""
Spatial cross-validation for landslide susceptibility/hazard models.

Landslide data is spatially autocorrelated — a random k-fold split leaks
information between train/test (nearby points end up on both sides), which
overstates reported performance. This module groups points into spatial
blocks (grid cells) and does leave-one-block-out CV instead.

Requirements:
    pip install numpy pandas scikit-learn

Usage:
    from spatial_cv import spatial_kfold_indices
    for train_idx, test_idx in spatial_kfold_indices(df, n_splits=5, block_size_km=20):
        ...
"""
import numpy as np
import pandas as pd


def _assign_blocks(df: pd.DataFrame, block_size_km: float) -> np.ndarray:
    """Assign each point to a spatial grid block based on lat/lon, at roughly
    block_size_km resolution (approximate — uses simple degree binning, fine
    for a regional-scale study area like Sikkim/NER)."""
    # ~1 degree latitude ~= 111 km
    deg_per_block = block_size_km / 111.0
    block_lat = (df["latitude"] // deg_per_block).astype(int)
    block_lon = (df["longitude"] // deg_per_block).astype(int)
    return (block_lat.astype(str) + "_" + block_lon.astype(str)).values


def spatial_kfold_indices(df: pd.DataFrame, n_splits: int = 5, block_size_km: float = 20,
                           seed: int = 42):
    """Yields (train_idx, test_idx) pairs, splitting by spatial block rather
    than by individual point — this is the leave-one-region-out style CV
    called for in the project plan's Model Validation Protocol."""
    blocks = _assign_blocks(df, block_size_km)
    unique_blocks = np.unique(blocks)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    folds = np.array_split(unique_blocks, n_splits)

    for fold_blocks in folds:
        test_mask = np.isin(blocks, fold_blocks)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "latitude": rng.uniform(27.0, 28.13, n),
        "longitude": rng.uniform(88.0, 88.93, n),
    })
    for i, (tr, te) in enumerate(spatial_kfold_indices(df, n_splits=5, block_size_km=20)):
        print(f"Fold {i}: train={len(tr)}, test={len(te)}")
