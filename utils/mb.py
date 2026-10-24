from .CI_test import CausalLearnCITest

import pandas as pd
import numpy as np
from typing import Union, List

class gaussian_MB_learn:
    """
    Implementation of the Gaussian Markov Blanket algorithm.
    References:
        - https://github.com/ban-epfl/rcd/blob/main/rcd/utilities/utils.py#L101
    """
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], alpha: float = None, **kwargs):
        from scipy import stats
        self.num_samples, self.num_nodes = data.shape
        if self.num_samples <= self.num_nodes:
            raise ValueError("Number of samples must be greater than number of nodes.")
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
            self.column_names = list(data.columns)

        elif isinstance(data, np.ndarray):
            self.data = data
            self.column_names = list(range(data.shape[1]))
        else:
            raise ValueError("Unsupported data type. Please provide a pandas DataFrame or a numpy array.")

        crr = np.corrcoef(self.data, rowvar=False)
        prec = np.linalg.pinv(crr)
        norm_vec = np.sqrt(np.diag(prec))
        mb_mat = np.abs(prec / norm_vec[:, None] / norm_vec[None, :])

        sig_level = 1 / self.num_nodes ** 2 if alpha is None else alpha

        thresh = np.tanh(stats.norm.ppf(1 - sig_level / 2) / np.sqrt(self.num_samples - self.num_nodes - 1))

        mb_mat = np.where(mb_mat > thresh, 1, -1)  # 1 means MB, -1 means not MB
        # set diagonal to 0
        np.fill_diagonal(mb_mat, 0)

        self.bool_mb_df = pd.DataFrame(
            mb_mat, index=self.column_names, columns=self.column_names
        )


    def __call__(self, target: Union[int, str], **kwargs) -> List[Union[int, str]]:
        """
        Get the Markov blanket of the target node.

        :param target: Target node (column label or index).
        :return: List of nodes in the Markov blanket.
        """
        mb = [col for col in self.bool_mb_df.columns if self.bool_mb_df.loc[target, col] == 1]
        return mb


class discrete_MB_learn:
    """Grow-Shrink Markov blanket learner using a discrete CI test."""

    def __init__(self, data, ci_test, **kwargs):
        self.columns = list(data.columns)
        self.ci_test = ci_test
        self.max_conditioning_size = kwargs.get("max_conditioning_size")

    def _limited_condition(self, nodes):
        if self.max_conditioning_size is None:
            return list(nodes)
        return list(nodes)[-int(self.max_conditioning_size):]

    def __call__(self, target, **kwargs):
        blanket = []
        changed = True
        while changed:
            changed = False
            for candidate in self.columns:
                if candidate == target or candidate in blanket:
                    continue
                independent, _ = self.ci_test(
                    target, candidate, self._limited_condition(blanket)
                )
                if not independent:
                    blanket.append(candidate)
                    changed = True
        for candidate in list(blanket):
            condition = self._limited_condition(
                [node for node in blanket if node != candidate]
            )
            independent, _ = self.ci_test(target, candidate, condition)
            if independent:
                blanket.remove(candidate)
        return blanket


class FastIAMB_learn:
    """Speculative Fast-IAMB using the project's shared CI-test cache."""

    def __init__(
        self,
        data,
        ci_test,
        alpha=0.05,
        max_k=3,
        data_type="discrete",
        max_iter=100,
        min_expected_per_cell=1,
        skip_shrink=False,
    ):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        self.data = data
        self.columns = list(data.columns)
        self.ci_test = ci_test
        self.alpha = float(alpha)
        self.max_k = max_k
        self.data_type = data_type
        self.max_iter = int(max_iter)
        self.min_expected_per_cell = float(min_expected_per_cell)
        self.skip_shrink = bool(skip_shrink)
        self.n_samples = data.shape[0]
        self.mb_set = {}

    def _enough_data(self, condition):
        if self.max_k is not None and len(condition) > self.max_k:
            return False
        if self.data_type == "discrete":
            required = self.min_expected_per_cell * 4 * (2 ** len(condition))
            return self.n_samples >= required
        return True

    def __call__(self, target, **kwargs):
        if target in self.mb_set:
            return list(self.mb_set[target])
        if target not in self.columns:
            raise ValueError(f"Target {target} is not in the observed variables.")

        candidates = [node for node in self.columns if node != target]
        blanket = []

        def dependent_candidates():
            if not self._enough_data(blanket):
                return []
            dependent = []
            for node in candidates:
                independent, p_value = self.ci_test(target, node, blanket)
                if not independent:
                    dependent.append((p_value, node))
            return sorted(dependent)

        dependent = dependent_candidates()
        iteration = 0
        while dependent and iteration < self.max_iter:
            iteration += 1
            insufficient_data = False
            for _, node in dependent:
                if not self._enough_data(blanket):
                    insufficient_data = True
                    break
                if node in candidates:
                    blanket.append(node)
                    candidates.remove(node)

            removed_any = False
            if not self.skip_shrink:
                for node in list(blanket):
                    condition = [item for item in blanket if item != node]
                    if not self._enough_data(condition):
                        continue
                    independent, _ = self.ci_test(target, node, condition)
                    if independent:
                        blanket.remove(node)
                        candidates.append(node)
                        removed_any = True

            if insufficient_data and (self.skip_shrink or not removed_any):
                break
            dependent = dependent_candidates()

        self.mb_set[target] = list(blanket)
        return list(blanket)

class TC_learn:
    """
    References:
        - Pellet J P, Elisseeff A. Using markov blankets for causal structure learning[J]. Journal of Machine Learning Research, 2008, 9(7).
    """
    def __init__(
        self,
        dag: pd.DataFrame,
        latent_vars: List[str]
    ):
        """
        :param dag: Fully correct 0/1 DAG adjacency matrix, which must be a pd.DataFrame.
        :param latent_vars: Set of latent variables, which must be a list[str].
        """
        if not isinstance(dag, pd.DataFrame):
            raise ValueError("dag must be a pandas DataFrame.")

        if dag.shape[0] != dag.shape[1]:
            raise ValueError("dag must be a square adjacency matrix.")

        if list(dag.index) != list(dag.columns):
            raise ValueError("For dag, index and columns must be identical.")

        if not isinstance(latent_vars, list) or not all(isinstance(v, str) for v in latent_vars):
            raise ValueError("latent_vars must be a list of strings.")

        if not np.isin(dag.to_numpy(), [0, 1]).all():
            raise ValueError("dag must be a 0/1 adjacency matrix.")

        self.dag = dag.copy()
        self.column_names = list(dag.columns)

        unknown_latent = set(latent_vars) - set(self.column_names)
        if unknown_latent:
            raise ValueError(f"Unknown latent variables: {sorted(unknown_latent)}")

        self.latent_vars = set(latent_vars)
        self.observed_variables = [var for var in self.column_names if var not in self.latent_vars]
        self.mb_set = {}

        self.ci_test = CausalLearnCITest(self.dag, method="d_sep", use_cache=False)

    def __call__(self, target: str, **kwargs) -> List[str]:
        """
        :param target: Target node name.
        :return: List of nodes in the Markov blanket.
        """
        if not isinstance(target, str):
            raise ValueError("target must be a string.")

        if target in self.latent_vars:
            raise ValueError(f"Target {target} is a latent variable, not an observed variable.")

        if target not in self.observed_variables:
            raise ValueError(f"Target {target} is not in the observed variables.")

        if target in self.mb_set:
            return self.mb_set[target]

        candidates = self.observed_variables.copy()
        candidates.remove(target)

        mb = []
        for y in candidates:
            cond_set = [var for var in candidates if var != y]

            # Convention: self.ci_test(X, Y, S)[0] == True means X is independent of Y given S.
            test_result = self.ci_test(target, y, cond_set)
            is_independent = test_result[0] if isinstance(test_result, (tuple, list)) else test_result

            if not is_independent:
                mb.append(y)

        self.mb_set[target] = mb
        return mb
