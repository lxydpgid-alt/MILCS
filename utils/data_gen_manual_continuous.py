import json
import os
from typing import Optional

import igraph as ig
import numpy as np
import pandas as pd

def mark_latents_and_sample_interventions(
    dag: pd.DataFrame,
    L: int,
    Is: int,
    I: int,
    seed: int | None = None,
    forbid_latent_nodes: list[str] | None = None,
    prefer_intervention_near: list[str] | None = None,
    common_intersection_size: int = 0,
    required_intervention_node: str | None = None,
    required_intervention_count: int | None = None,
) -> tuple[pd.DataFrame, list[set[str]]]:
    rng = np.random.default_rng(seed)

    forbid_latent_nodes = [] if forbid_latent_nodes is None else [str(x) for x in forbid_latent_nodes]
    forbidden = set(forbid_latent_nodes)
    multi_child_nodes = [
        node
        for node in dag.index[dag.sum(axis=1) >= 2].to_list()
        if str(node) not in forbidden
    ]
    root_multi_child_nodes = [
        node for node in multi_child_nodes if dag.loc[:, node].sum() == 0
    ]
    remaining_multi_child_nodes = [
        node for node in multi_child_nodes if node not in root_multi_child_nodes
    ]

    latent_count = min(L, len(multi_child_nodes))
    take_root = min(latent_count, len(root_multi_child_nodes))
    latent_old = (
        rng.choice(root_multi_child_nodes, size=take_root, replace=False).tolist()
        if take_root > 0 else []
    )
    take_remaining = latent_count - len(latent_old)
    if take_remaining > 0:
        latent_old.extend(
            rng.choice(
                remaining_multi_child_nodes,
                size=take_remaining,
                replace=False,
            ).tolist()
        )
    latent_new = [f"L{i}" for i in range(1, latent_count + 1)]
    rename_map = dict(zip(latent_old, latent_new))
    dag = dag.rename(index=rename_map, columns=rename_map)

    observed = [v for v in dag.columns if not v.startswith("L")]

    prefer_intervention_near = [] if prefer_intervention_near is None else [str(x) for x in prefer_intervention_near]
    preferred_targets = [x for x in prefer_intervention_near if x in observed]
    preferred_nodes: list[str] = []
    if preferred_targets:
        pref_set = set()
        for t in preferred_targets:
            pa = set(dag.index[dag.loc[:, t] > 0].tolist())
            ch = set(dag.columns[dag.loc[t, :] > 0].tolist())
            neigh = {x for x in (pa | ch) if x in observed and x != t}
            pref_set.update(neigh)
        preferred_nodes = sorted(pref_set)
    fallback_nodes = [x for x in observed if x not in preferred_nodes]

    def sample_one_intervention_set() -> set[str]:
        if len(observed) <= I:
            return set(observed)
        picks: list[str] = []
        if preferred_nodes:
            n_pref = min(max(1, I - 1), len(preferred_nodes))
            picks.extend(rng.choice(preferred_nodes, size=n_pref, replace=False).tolist())
        remaining = I - len(picks)
        if remaining > 0:
            pool = [x for x in fallback_nodes if x not in picks]
            if len(pool) < remaining:
                pool = [x for x in observed if x not in picks]
            picks.extend(rng.choice(pool, size=remaining, replace=False).tolist())
        return set(picks)

    if common_intersection_size < 0:
        raise ValueError("common_intersection_size must be >= 0")
    if common_intersection_size > I:
        raise ValueError("common_intersection_size cannot exceed I")

    preferred_common_pool = [x for x in preferred_nodes if x in observed]
    fallback_common_pool = [x for x in observed if x not in preferred_common_pool]

    def sample_with_required_intersection() -> list[set[str]]:
        if common_intersection_size == 0:
            while True:
                intervention_sets_local = [sample_one_intervention_set() for _ in range(Is)]
                if not set.intersection(*intervention_sets_local):
                    return intervention_sets_local

        if len(observed) < I:
            raise ValueError("not enough observed nodes for requested intervention size")

        common_nodes: list[str] = []
        if common_intersection_size > 0:
            take_pref = min(common_intersection_size, len(preferred_common_pool))
            if take_pref > 0:
                common_nodes.extend(rng.choice(preferred_common_pool, size=take_pref, replace=False).tolist())
            remaining_common = common_intersection_size - len(common_nodes)
            if remaining_common > 0:
                pool = [x for x in fallback_common_pool if x not in common_nodes]
                if len(pool) < remaining_common:
                    pool = [x for x in observed if x not in common_nodes]
                common_nodes.extend(rng.choice(pool, size=remaining_common, replace=False).tolist())

        common_set = set(common_nodes)
        if len(common_set) != common_intersection_size:
            raise ValueError("failed to sample the requested common intersection")

        intervention_sets_local: list[set[str]] = []
        extra_size = I - common_intersection_size
        available_for_unique = [x for x in observed if x not in common_set]
        if extra_size > 0 and len(available_for_unique) < extra_size * Is:
            raise ValueError("not enough remaining observed nodes to enforce exact intersection size")

        used_unique: set[str] = set()
        for _ in range(Is):
            picks = set(common_set)
            if extra_size > 0:
                preferred_unique_pool = [
                    x for x in preferred_nodes
                    if x in available_for_unique and x not in used_unique
                ]
                fallback_unique_pool = [
                    x for x in available_for_unique
                    if x not in used_unique and x not in preferred_unique_pool
                ]
                unique_picks: list[str] = []
                take_pref = min(extra_size, len(preferred_unique_pool))
                if take_pref > 0:
                    unique_picks.extend(rng.choice(preferred_unique_pool, size=take_pref, replace=False).tolist())
                remaining = extra_size - len(unique_picks)
                if remaining > 0:
                    pool = [x for x in fallback_unique_pool if x not in unique_picks]
                    if len(pool) < remaining:
                        pool = [x for x in available_for_unique if x not in used_unique and x not in unique_picks]
                    unique_picks.extend(rng.choice(pool, size=remaining, replace=False).tolist())
                picks.update(unique_picks)
                used_unique.update(unique_picks)
            intervention_sets_local.append(picks)

        exact_intersection = set.intersection(*intervention_sets_local) if intervention_sets_local else set()
        if len(exact_intersection) != common_intersection_size:
            raise RuntimeError("failed to construct intervention sets with exact requested intersection size")
        return intervention_sets_local

    if required_intervention_count is not None:
        if required_intervention_node not in observed:
            raise ValueError(
                f"required intervention node {required_intervention_node!r} is not observed"
            )
        if not 0 <= required_intervention_count <= Is:
            raise ValueError("required_intervention_count must be between 0 and Is")

    def enforce_required_count(sets: list[set[str]]) -> list[set[str]]:
        if required_intervention_count is None:
            return sets
        node = str(required_intervention_node)
        selected = set(
            rng.choice(Is, size=required_intervention_count, replace=False).tolist()
        )
        adjusted = [set(s) for s in sets]
        for env_idx, intervention_set in enumerate(adjusted):
            should_include = env_idx in selected
            if should_include and node not in intervention_set:
                removable = sorted(intervention_set - {node})
                intervention_set.remove(str(rng.choice(removable)))
                intervention_set.add(node)
            elif not should_include and node in intervention_set:
                intervention_set.remove(node)
                replacements = sorted(set(observed) - intervention_set - {node})
                intervention_set.add(str(rng.choice(replacements)))
        return adjusted

    for _ in range(1000):
        intervention_sets = enforce_required_count(sample_with_required_intersection())
        exact_intersection = set.intersection(*intervention_sets) if intervention_sets else set()
        actual_count = (
            sum(required_intervention_node in s for s in intervention_sets)
            if required_intervention_count is not None else None
        )
        if (
            len(exact_intersection) == common_intersection_size
            and (required_intervention_count is None or actual_count == required_intervention_count)
        ):
            break
    else:
        raise RuntimeError("failed to construct intervention sets with required target count")
    return dag, intervention_sets
 # Return the renamed DAG together with the list of intervention sets.


def simulate_linear_sem(
    W,
    n,
    noise_scale=None,
    intervention_nodes=None,
):
    def _simulate_single_equation(X, w, scale):
        z = np.random.normal(scale=scale, size=n)#guass
        x = X @ w + z
        return x
    # If noise_scale is None, every node uses noise standard deviation 1.
    # If noise_scale is a scalar such as 0.5, all nodes use 0.5.
    # If noise_scale is a length-d vector, each node can use its own noise scale.
    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        scale_vec = noise_scale

    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, d])
    intervention_nodes = set() if intervention_nodes is None else set(intervention_nodes)
    for j in ordered_vertices:
        if j in intervention_nodes:
            # Keep only zero-mean noise, while reusing the same noise scale.
            X[:, j] = np.random.normal(scale=scale_vec[j], size=n)# replaceable
        else:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


FUNCTIONS = {
    "cubic": lambda x: 0.8 * x - 0.4 * x**2 + 0.2 * x**3,
    "exp": lambda x: np.exp(0.3 * x),
    "tanh": lambda x: np.tanh(x),
    "softplus": lambda x: np.log1p(np.exp(np.clip(x, -20, 20))),
}
FUNCTION_NAMES = tuple(FUNCTIONS)


def _sample_nonlinear_function_ids(W, seed):
    """Choose one reproducible nonlinear function independently per edge."""
    rng = np.random.default_rng(seed)
    function_ids = np.full(W.shape, -1, dtype=np.int8)
    edges = np.argwhere(W != 0)
    if len(edges):
        choices = rng.integers(0, len(FUNCTIONS), size=len(edges), dtype=np.int8)
        function_ids[edges[:, 0], edges[:, 1]] = choices
    return function_ids


def simulate_square_sem(
    W,
    n,
    noise_scale=1.0,
    intervention_nodes=None,
    nonlinear_function_seed=0,
):
    """Continuous additive-noise SEM with a random nonlinear function per edge."""
    d = W.shape[0]
    scale_vec = np.full(d, noise_scale, dtype=float)
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    X = np.zeros((n, d))
    intervention_nodes = set() if intervention_nodes is None else set(intervention_nodes)
    function_ids = _sample_nonlinear_function_ids(W, nonlinear_function_seed)
    for j in G.topological_sorting():
        noise = np.random.normal(scale=scale_vec[j], size=n)
        if j in intervention_nodes:
            X[:, j] = noise
            continue
        parents = G.neighbors(j, mode=ig.IN)
        value = np.zeros(n)
        for parent in parents:
            function_name = FUNCTION_NAMES[function_ids[parent, j]]
            value += W[parent, j] * FUNCTIONS[function_name](X[:, parent])
        X[:, j] = value + noise
        # Prevent large values from dominating later squared terms.
        std = X[:, j].std()
        if std > 1e-8:
            X[:, j] = (X[:, j] - X[:, j].mean()) / std
    return X


def simulate_discrete_hard_sem(
    W,
    n,
    intervention_nodes=None,
    intervention_probability=0.5,
):
    """Binary logistic SEM with perfect (hard) interventions."""
    if not 0.0 < intervention_probability < 1.0:
        raise ValueError("intervention_probability must be between 0 and 1")
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    X = np.zeros((n, W.shape[0]), dtype=int)
    intervention_nodes = set() if intervention_nodes is None else set(intervention_nodes)
    for j in G.topological_sorting():
        if j in intervention_nodes:
            # do(X_j): remove all parent effects and sample from a fixed external distribution.
            probability = np.full(n, intervention_probability)
        else:
            parents = G.neighbors(j, mode=ig.IN)
            logits = X[:, parents] @ W[parents, j] if parents else np.zeros(n)
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        X[:, j] = np.random.binomial(1, probability, size=n)
    return X


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def build_design(
    dag_matrix_path: str,
    L: int = 3,
    Is: int = 3,# number of interventional environments
    I: int = 3,# number of intervention nodes per environment
    seed: Optional[int] = None,
    forbid_latent_nodes: Optional[list[str]] = None,
    prefer_intervention_near: Optional[list[str]] = None,
    common_intersection_size: int = 0,
    required_intervention_node: Optional[str] = None,
    required_intervention_count: Optional[int] = None,
):
    if seed is not None:
        np.random.seed(seed)

    dag_df = pd.read_csv(os.path.expanduser(dag_matrix_path), index_col=0)
    dag_df = (dag_df > 0).astype(int)
    dag_df.index = dag_df.index.map(str)
    dag_df.columns = dag_df.columns.map(str)

    dag_df, intervention_sets = mark_latents_and_sample_interventions(
        dag=dag_df,
        L=L,
        Is=Is,
        I=I,
        seed=seed,
        forbid_latent_nodes=forbid_latent_nodes,
        prefer_intervention_near=prefer_intervention_near,
        common_intersection_size=common_intersection_size,
        required_intervention_node=required_intervention_node,
        required_intervention_count=required_intervention_count,
    )

    labels = dag_df.columns.to_list()
    A_dag = dag_df.to_numpy(dtype=int)
    observed_labels = [name for name in labels if not name.startswith("L")]
    interventions = [sorted(s) for s in intervention_sets] + [[]]

    if seed is not None:
        np.random.seed(seed)
    W_base = simulate_parameter(A_dag)

    env_matrices = []
    for intervention in interventions:
        A_do_df = dag_df.copy()
        for node in intervention:
            A_do_df.loc[:, node] = 0
        env_matrices.append(A_do_df)

    return {
        "truth_dag_df": dag_df,
        "env_matrices": env_matrices,
        "observed_labels": observed_labels,
        "interventions": interventions,
        "W_base": W_base,
        "seed": seed,
        "L": L,
        "Is": Is,
        "I": I,
        "dag_matrix_path": os.path.expanduser(dag_matrix_path),
        "forbid_latent_nodes": [] if forbid_latent_nodes is None else list(forbid_latent_nodes),
        "prefer_intervention_near": [] if prefer_intervention_near is None else list(prefer_intervention_near),
        "common_intersection_size": common_intersection_size,
        "required_intervention_node": required_intervention_node,
        "required_intervention_count": required_intervention_count,
    }






def generate_data_from_design(
    design: dict,
    intervention_dir: str,
    n_samples: int = 2000,
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
    data_model: str = "linear",
    discrete_intervention_probability: float = 0.5,
):
    os.makedirs(intervention_dir, exist_ok=True)

    truth_dag_df = design["truth_dag_df"]
    labels = truth_dag_df.columns.to_list()
    observed_labels = design["observed_labels"]
    observed_indices = [labels.index(name) for name in observed_labels]
    interventions = design["interventions"]
    W_base = design["W_base"]

    all_data_paths = []
    for idx, intervention in enumerate(interventions, start=1):
        if idx == len(interventions):
            continue

        if seed is not None:
            np.random.seed(seed + idx * 1009)

        intervention_idx = [labels.index(node) for node in intervention]
        if data_model == "discrete":
            data = simulate_discrete_hard_sem(
                W=W_base,
                n=n_samples,
                intervention_nodes=intervention_idx,
                intervention_probability=discrete_intervention_probability,
            )
        elif data_model == "square":
            data = simulate_square_sem(
                W=W_base,
                n=n_samples,
                noise_scale=noise_scale,
                intervention_nodes=intervention_idx,
                nonlinear_function_seed=0 if seed is None else seed,
            )
        else:
            data = simulate_linear_sem(
                W=W_base,
                n=n_samples,
                noise_scale=noise_scale,
                intervention_nodes=intervention_idx,
            )

        data = data[:, observed_indices]
        out_path = os.path.join(intervention_dir, f"data_{idx}.csv")
        pd.DataFrame(data, columns=observed_labels).to_csv(out_path, index=False)
        all_data_paths.append(out_path)

    return {"data_paths": all_data_paths}
