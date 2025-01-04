import seaborn as sns
from colorsys import rgb_to_yiq
from typing import Dict, List


def convert_color_dict_to_grayscale(
        color_dict: Dict,
        ignore_keys: List = None,
        lighten: float = 0.5,
) -> Dict:
    if ignore_keys is None:
        ignore_keys = []
    graymap = sns.color_palette('gray', as_cmap=True)
    gray_dict = dict()
    for k, v in color_dict.items():
        if k in ignore_keys:
            gray_dict[k] = v
        else:
            y_ = rgb_to_yiq(*v)[0]
            y_scale = (1 - lighten*(1 - y_))
            gray_dict[k] = graymap(y_scale)
    return gray_dict
