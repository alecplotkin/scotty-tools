import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Dict


def get_subset_frequency_table(
        df: pd.DataFrame,
        day_field: str = 'day',
        subset_field: str = 'subset',
):
    df_counts = df.value_counts([day_field, subset_field]).reset_index()
    df_counts['total'] = df_counts.groupby(day_field)['count'].transform('sum')
    df_counts['frequency'] = df_counts['count'] / df_counts['total']
    return df_counts


def plot_subset_frequencies(
        df: pd.DataFrame,
        day_field: str = 'day',
        subset_field: str = 'subset',
        plot_field: Literal['count', 'frequency'] = 'frequency',
        time_as_categorical: bool = True,
        color_dict: Dict = None
):
    df_freq = get_subset_frequency_table(df, day_field, subset_field)
    df_freq = df_freq.sort_values(day_field)
    if time_as_categorical:
        ix_field = day_field + '_ix'
        df_freq[ix_field] = pd.factorize(df_freq[day_field])[0]
        plt.xticks(
            ticks=df_freq[ix_field].unique(),
            labels=df_freq[day_field].unique(),
        )
        day_field = ix_field
    for subset, df in df_freq.groupby(subset_field):
        if color_dict is not None:
            c = color_dict[subset]
        else:
            c = None
        plt.plot(df[day_field], df[plot_field], c=c, label=subset)
    plt.legend()
    return
