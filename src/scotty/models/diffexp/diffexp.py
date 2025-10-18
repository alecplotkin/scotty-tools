import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class TemporalDifferentialExpression:
    def __init__(self, df, gene, group_var, time_var):
        self.df = df
        self.gene = gene
        self.group_var = group_var
        self.time_var = time_var
        self.model = smf.ols(f'{gene} ~ C({group_var}) * C({time_var})', data=df)
        self._is_fitted = False
        self.params = None
        self.param_names = None

    def fit(self):
        self.model = self.model.fit()
        self._is_fitted = True
        self.params = self.model.params
        self.param_names = self.model.params.index

    def _get_param_idx(self, term):
        try:
            return self.param_names.get_loc(term)
        except KeyError:
            return None

    # Constructs and runs time-averaged t-test for 'group' vs other groups in group_var
    def time_averaged_t_test(self, group):
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before running t-test.")

        group_var = self.group_var
        time_var = self.time_var
        group_levels = sorted(self.df[group_var].unique())
        time_levels = sorted(self.df[time_var].unique())

        df_x = self.df.loc[self.df[group_var] != group, :]
        proportions = df_x.groupby(time_var)[group_var].value_counts(normalize=True)

        other_groups = [g for g in group_levels if g != group]

        contrast = np.zeros(len(self.params))
        weight = 1 / len(time_levels)

        main_effect_test = f'C({group_var})[T.{group}]'
        idx_main_others = dict()
        for g in other_groups:
            idx_main_others[g] = self._get_param_idx(f'C({group_var})[T.{g}]')

        for t in time_levels:
            interaction_test = f'C({group_var})[T.{group}]:C({time_var})[T.{t}]'
            idx_inter_others = dict()
            for g in other_groups:
                inter_other = f'C({group_var})[T.{g}]:C({time_var})[T.{t}]'
                idx_inter_others[g] = self._get_param_idx(inter_other)

            idx_main_test = self._get_param_idx(main_effect_test)
            if idx_main_test is not None:
                contrast[idx_main_test] += weight

            for g, idx_main_other in idx_main_others.items():
                if idx_main_other is not None:
                    contrast[idx_main_other] -= weight * proportions[(t, g)]

            idx_inter_test = self._get_param_idx(interaction_test)
            if idx_inter_test is not None:
                contrast[idx_inter_test] += weight

            for g, idx_inter_other in idx_inter_others.items():
                if idx_inter_other is not None:
                    contrast[idx_inter_other] -= weight * proportions[(t, g)]

        return self.model.t_test(contrast)

    def test_all_groups(self) -> pd.DataFrame:
        group_levels = sorted(self.df[self.group_var].unique())
        res = dict()
        for group in group_levels:
            res[group] = self.time_averaged_t_test(group).summary_frame()
        res = pd.concat(res.values(), keys=res.keys(), names=['group'])
        return res.droplevel(1, axis=0)


def _worker(df: pd.DataFrame, feat: str, group_var: str, time_var: str) -> tuple[str, pd.DataFrame]:
    tde = TemporalDifferentialExpression(df, feat, group_var, time_var)
    tde.fit()
    result = tde.test_all_groups()
    return feat, result


def calculate_temporal_diff_exp(
    df: pd.DataFrame,
    feature_vars: list[str],
    group_var: str,
    time_var: str,
    max_workers: int = 8,
) -> pd.DataFrame:
    results = dict()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for feat in feature_vars:
            sub_df = df[[feat, group_var, time_var]].copy()
            futures.append(executor.submit(_worker, sub_df, feat, group_var, time_var))
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    feat, result = future.result()
                    results[feat] = result
                except Exception as e:
                    print(f'{feat} generated an exception: {e}')
                pbar.update(1)
    return pd.concat(results.values(), keys=results.keys(), names=['names'])
