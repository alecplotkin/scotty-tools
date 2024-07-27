from typing import Tuple, List, Iterable


def window(x: Iterable, k: int = 2) -> List[Tuple]:
    windows = []
    for i in range(len(x)-k+1):
        window = tuple((x[i+j] for j in range(k)))
        windows.append(window)
    return windows
