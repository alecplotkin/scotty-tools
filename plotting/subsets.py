import numpy as np
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


def plot_subset_frequencies_trajectory(freqs, sub, c=None, ax=None):
    if ax is None:
        ax = plt.gca()
    actual_freq = freqs.loc[freqs['ref_time'] == freqs['day'], sub].to_numpy()
    ax.plot(actual_freq, c=c)
    timepoints = freqs['ref_time'].unique()
    for time in timepoints:
        pred_freq = freqs.loc[freqs['ref_time'] == time, sub].to_numpy()
        if np.allclose(pred_freq, 0):
            continue
        pad = np.full(len(actual_freq) - len(pred_freq), np.nan)
        pred_freq = np.concatenate((pad, pred_freq))
        ax.plot(pred_freq, linestyle=':', c=c)
    ax.set_xticks(
        ticks=np.arange(len(timepoints)),
        labels=timepoints,
        rotation=90
    )
    return ax
