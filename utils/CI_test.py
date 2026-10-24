import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from typing import Union, List, Optional, Dict, Tuple
from causallearn.utils.cit import CIT


def _force_fastkci_single_process() -> None:
    """Avoid FastKCI's hard-coded ``n_jobs=-1`` process pool.

    causal-learn currently creates a loky process pool for every FastKCI call.
    Besides being expensive for the small block jobs, this fails in restricted
    environments where POSIX semaphore limits cannot be queried.  Keep the
    algorithm unchanged but execute its joblib tasks sequentially.
    """
    import importlib
    from joblib import Parallel as JoblibParallel

    fastkci_module = importlib.import_module("causallearn.utils.FastKCI.FastKCI")
    if getattr(fastkci_module, "_milcs_single_process", False):
        return

    def single_process_parallel(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return JoblibParallel(*args, **kwargs)

    fastkci_module.Parallel = single_process_parallel
    fastkci_module._milcs_single_process = True


class CausalLearnCITest:
    """
    Lightweight wrapper around causal-learn CIT that only accepts pd.DataFrame.
    Supports:
    - "fisherz" / "mv_fisherz" / "kci" : continuous data
    - "gsq" / "chisq"                  : discrete data

    Usage:
        cit = CausalLearnCITest(df, method="fisherz", alpha=0.01)
        CI, p = cit("X", "Y", ["Z1", "Z2"])
    """

    DISCRETE_METHODS = {"gsq", "chisq"}
    CONTINUOUS_METHODS = {"fisherz", "mv_fisherz", "kci", "rcit", "fastkci", "polyz"}
    SUPPORTED_METHODS = DISCRETE_METHODS | CONTINUOUS_METHODS

    def __init__(
        self,
        data: pd.DataFrame,
        method: str = "fisherz",
        alpha: float = 0.05,
        Max_time: Optional[float] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CausalLearnCITest only accepts pd.DataFrame.")

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods are: {sorted(self.SUPPORTED_METHODS)}"
            )

        if data.columns.duplicated().any():
            raise ValueError("DataFrame columns must be unique.")

        self.alpha = float(alpha)
        self.method = method
        self.Max_time = Max_time
        self.start_time = time.process_time()
        self.use_cache = bool(use_cache)
        self._cache: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
        self._ci_num = 0

        self.df = data.copy()
        self.columns = list(map(str, self.df.columns))
        self.df.columns = self.columns
        self.col_to_idx = {col: idx for idx, col in enumerate(self.columns)}

        self.data = self._prepare_data(self.df)
        if self.method == "polyz":
            # Cache correlations of [X_1,...,X_p,X_1^2,...,X_p^2] once.
            # CI queries then operate only on small covariance submatrices.
            centered = self.data - self.data.mean(axis=0, keepdims=True)
            squared = self.data * self.data
            squared -= squared.mean(axis=0, keepdims=True)
            poly = np.column_stack((centered, squared))
            scale = poly.std(axis=0, ddof=1)
            scale[scale <= 1e-12] = 1.0
            poly /= scale
            self._poly_corr = np.corrcoef(poly, rowvar=False)
            self._poly_p = self.data.shape[1]
        if self.method == "polyz":
            self.cit_obj = None
        else:
            if self.method == "fastkci":
                _force_fastkci_single_process()
            self.cit_obj = CIT(self.data, self.method, **kwargs)

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        if self.method in self.DISCRETE_METHODS:
            if df.isna().any().any():
                raise ValueError(f"Method '{self.method}' does not allow missing values.")

            return np.column_stack([
                pd.factorize(df[col], sort=True)[0].astype(np.int64, copy=False)
                for col in df.columns
            ])

        try:
            data = df.to_numpy(dtype=float, copy=True)
        except ValueError as exc:
            raise TypeError(
                f"Method '{self.method}' requires numeric continuous data."
            ) from exc

        if np.isinf(data).any():
            raise ValueError("Input data contains Inf.")

        if self.method != "mv_fisherz" and np.isnan(data).any():
            raise ValueError(
                f"Method '{self.method}' does not allow NaN. "
                f"Use 'mv_fisherz' if missing values are present."
            )

        return data

    def _to_idx(self, value: Union[int, str]) -> int:
        if isinstance(value, str):
            if value not in self.col_to_idx:
                raise ValueError(f"Unknown variable name: {value}")
            return self.col_to_idx[value]
        return int(value)

    def _format_query(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        S: Optional[List[Union[int, str]]] = None,
    ) -> Tuple[int, int, List[int]]:
        S = [] if S is None else sorted({self._to_idx(s) for s in S})
        X = self._to_idx(X)
        Y = self._to_idx(Y)

        if X == Y:
            raise ValueError("X and Y must be different.")
        if X in S or Y in S:
            raise ValueError("Conditioning set S must not contain X or Y.")

        return X, Y, S

    def _cache_key(self, X: int, Y: int, S: List[int]) -> Tuple[int, int, Tuple[int, ...]]:
        a, b = sorted((X, Y))
        return a, b, tuple(S)

    def _polyz_pvalue(self, X: int, Y: int, S: List[int]) -> float:
        """Quadratic partial-correlation test using a cached feature matrix."""
        p = self._poly_p
        condition_features = list(S) + [s + p for s in S]
        dof = max(self.data.shape[0] - len(condition_features) - 3, 1)
        p_values = []
        for x_feature in (X, X + p):
            for y_feature in (Y, Y + p):
                indices = [x_feature, y_feature] + condition_features
                precision = np.linalg.pinv(self._poly_corr[np.ix_(indices, indices)])
                denom = np.sqrt(max(precision[0, 0] * precision[1, 1], 1e-24))
                r = float(-precision[0, 1] / denom)
                r = np.clip(r, -1 + 1e-12, 1 - 1e-12)
                statistic = np.sqrt(dof) * np.arctanh(abs(r))
                p_values.append(float(2 * norm.sf(statistic)))
        return min(1.0, 4.0 * min(p_values))

    def get_ci_num(self) -> int:
        return self._ci_num

    def __call__(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        S: Optional[List[Union[int, str]]] = None,
        **kwargs,
    ) -> Tuple[bool, float]:
        if self.Max_time is not None and time.process_time() - self.start_time > self.Max_time:
            raise TimeoutError(
                f"The simulation exceeded maximum time limit of {self.Max_time} seconds."
            )

        X, Y, S = self._format_query(X, Y, S)
        key = self._cache_key(X, Y, S)

        if self.use_cache and key in self._cache:
            p_value = self._cache[key]
        else:
            p_value = (
                self._polyz_pvalue(X, Y, S)
                if self.method == "polyz"
                else float(self.cit_obj(X, Y, S))
            )
            if self.use_cache:
                self._cache[key] = p_value
            self._ci_num += 1

        CI = p_value > self.alpha
        dep = -np.log(max(p_value, 1e-300))

        if kwargs.get("output_dict", False):
            return {"CI": CI, "p_value": p_value, "dep": dep}
        return CI, p_value
