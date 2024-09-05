import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Iterable
from statsmodels.stats.multitest import multipletests


def window(x: Iterable, k: int = 2) -> List[Tuple]:
    windows = []
    for i in range(len(x)-k+1):
        window = tuple((x[i+j] for j in range(k)))
        windows.append(window)
    return windows


def adjust_pvalues(pvals: npt.NDArray, method: str = 'fdr_bh'):
    padj = np.full_like(pvals, np.nan)
    mask = np.isfinite(pvals)
    if mask.sum() != 0:
        padj[mask] = multipletests(pvals[mask], alpha=0.05, method=method)[1]
    return padj
