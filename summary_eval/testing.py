"""
This file establishes a common testing framework for evaluating
models we create. This way, all our results should be comparable.
"""
import statistics
from typing import List

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import  KFold
import pandas as pd
from tqdm.notebook import tqdm

from summary_eval.settings import N_FOLDS, N_RUNS
from summary_eval.util import logger


class CrossValidator:
    """
    We use a custom cross-validator because we want to
    calculate metrics for multiple metrics across multiple target columns
    and sklearn currently does not support this.
    """

    def __init__(self, model, X_train, y_train) -> None:
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.targets = y_train.columns

        # A dictionary of form:
        # { "metric1": { "target1": [value1, value2, ...], ... }, ... }
        self.metrics = {
            "rmse": {},
            "mae": {},
            "r2": {},
        }

    def cross_validate(self, n_folds: int, n_runs: int) -> pd.DataFrame:
        logger.info(f"Using {n_runs}x{n_folds} cross validation")
        # Generate a unique seed for each run, but for n runs,
        # generate the same set of seeds for reproducibility
        seeds = [i for i in range(n_runs)]
        with tqdm(total=n_runs * n_folds) as pbar:
            for run_i in range(n_runs):
                k_folds = KFold(n_splits=n_folds, shuffle=True, random_state=seeds[run_i])
                for train_i, test_i in k_folds.split(self.X_train):
                    fold_train_X, fold_test_X = self.X_train.iloc[train_i], self.X_train.iloc[test_i]
                    fold_train_y, fold_test_y = self.y_train.iloc[train_i], self.y_train.iloc[test_i]
                    self.model.fit(fold_train_X, fold_train_y)
                    y_pred = self.model.predict(fold_test_X)

                    # Calculate metrics
                    rmses = root_mean_squared_error(fold_test_y, y_pred, multioutput="raw_values")
                    self._update_metric(rmses, "rmse")
                    maes = mean_absolute_error(fold_test_y, y_pred, multioutput="raw_values")
                    self._update_metric(maes, "mae")
                    r2s = r2_score(fold_test_y, y_pred, multioutput="raw_values")
                    self._update_metric(r2s, "r2")
                    pbar.update(1)
        return self._calculate_cv_results()

    def _update_metric(self, results: List[List[float]], name: str) -> None:
        for i, fold_value in enumerate(results):
            fold_values = self.metrics[name].get(self.targets[i], [])
            self.metrics[name][self.targets[i]] = fold_values + [fold_value]

    def _calculate_cv_results(self) -> pd.DataFrame:
        results = []
        for metric, targets in self.metrics.items():
            means = []
            stdevs = []
            for target, values in targets.items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                means.append(mean)
                stdevs.append(stdev)
                results.append([metric, target, mean, stdev, len(values)])
            results.append([metric, "mean_columnwise", statistics.mean(means), statistics.mean(stdevs), len(means)])

        results_df = pd.DataFrame(results, columns=["metric", "target", "mean", "stdev", "n_trials"])
        metrics = results_df["metric"].unique().tolist()
        targets = results_df["target"].unique().tolist()
        results_df = results_df.transpose()
        results_df.columns = pd.MultiIndex.from_product([metrics,targets,], names=["Metric", "Target"])
        return results_df.iloc[2:]


def cross_validate(model, X_train, y_train, n_folds: int = N_FOLDS, n_runs: int = N_RUNS):
    cv = CrossValidator(model, X_train, y_train)
    return cv.cross_validate(n_folds, n_runs)
