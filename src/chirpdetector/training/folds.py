"""Grouped k-fold assignment: folds are made of whole groups.

A group (a simulated scene with all its variants, or a real recording) is
in exactly one fold, so no source is ever on both sides of a split (audit
N2). Folds are balanced by number of images: groups are taken from the
largest to the smallest (ties in a seeded random order) and each goes to
the fold with the fewest images so far.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["assign_folds", "fold_groups"]


def assign_folds(
    group_sizes: Mapping[str, int], n_folds: int, seed: int = 0
) -> dict[str, int]:
    """Assign every group to one of ``n_folds`` folds.

    Parameters
    ----------
    group_sizes
        Number of images per group.
    n_folds
        Number of folds, >= 2.
    seed
        Seed of the tie-breaking order.

    Returns
    -------
    dict
        Group -> fold index in ``0 .. n_folds - 1``; every fold non-empty.

    Raises
    ------
    ValueError
        If ``n_folds < 2`` or there are fewer groups than folds.
    """
    if n_folds < 2:
        msg = f"need at least 2 folds, got {n_folds}"
        raise ValueError(msg)
    groups = sorted(group_sizes)
    if len(groups) < n_folds:
        msg = f"{len(groups)} groups cannot fill {n_folds} folds"
        raise ValueError(msg)
    tiebreak = np.random.default_rng(seed).permutation(len(groups))
    order = sorted(
        range(len(groups)), key=lambda k: (-int(group_sizes[groups[k]]), tiebreak[k])
    )
    load = [0] * n_folds
    count = [0] * n_folds
    out: dict[str, int] = {}
    for k in order:
        fold = min(range(n_folds), key=lambda f: (load[f], count[f], f))
        out[groups[k]] = fold
        load[fold] += int(group_sizes[groups[k]])
        count[fold] += 1
    return out


def fold_groups(
    assignment: Mapping[str, int], fold: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Training and validation groups of one fold, sorted."""
    if fold not in set(assignment.values()):
        msg = f"no fold {fold}; folds are {sorted(set(assignment.values()))}"
        raise ValueError(msg)
    train = tuple(sorted(g for g, f in assignment.items() if f != fold))
    val = tuple(sorted(g for g, f in assignment.items() if f == fold))
    return train, val
