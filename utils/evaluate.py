"""Matrix conversion and target-neighborhood evaluation for MILCS."""

import pandas as pd

from Graph.mixgraph import MixGraph, Mark


# Build matrices within the target-neighborhood scope:
# convert the predicted graph G into pred_smcm and the ground truth into true_smcm.
def build_eval_matrices(
    G: MixGraph,
    truth_matrix_path: str,
    latent_vars,
):
    truth_df = pd.read_csv(truth_matrix_path, index_col=0)
    all_labels = list(truth_df.columns)
    A = truth_df.values

    latent_set = set(latent_vars) & set(all_labels)
    labels = [name for name in all_labels if name not in latent_set]

    pred_smcm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, _, attr in G.edges(data=True):
        edge = attr["edge"]
        u = edge.start.name
        v = edge.end.name
        if u not in pred_smcm.index or v not in pred_smcm.columns:
            continue
        lmark = edge.lmark
        rmark = edge.rmark
        if lmark == Mark.CIRCLE and rmark == Mark.CIRCLE:
            pred_smcm.loc[u, v] = 1
            pred_smcm.loc[v, u] = 1
        elif lmark == Mark.TAIL and rmark == Mark.ARROW:
            pred_smcm.loc[u, v] = 2
            pred_smcm.loc[v, u] = 3
        elif lmark == Mark.ARROW and rmark == Mark.TAIL:
            pred_smcm.loc[u, v] = 3
            pred_smcm.loc[v, u] = 2
        elif lmark == Mark.ARROW and rmark == Mark.ARROW:
            pred_smcm.loc[u, v] = 2
            pred_smcm.loc[v, u] = 2
        elif lmark == Mark.CIRCLE and rmark == Mark.ARROW:
            pred_smcm.loc[u, v] = 4
            pred_smcm.loc[v, u] = 5
        elif lmark == Mark.ARROW and rmark == Mark.CIRCLE:
            pred_smcm.loc[u, v] = 5
            pred_smcm.loc[v, u] = 4

    true_smcm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    latent_nodes = [all_labels.index(v) for v in latent_set if v in all_labels]

    def has_latent(i: int, j: int) -> bool:
        for L in latent_nodes:
            if A[L, i] == 1 and A[L, j] == 1:
                return True
        return False

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            x_name = labels[i]
            y_name = labels[j]
            x = all_labels.index(x_name)
            y = all_labels.index(y_name)

            direct_xy = A[x, y] == 1
            direct_yx = A[y, x] == 1
            latent_xy = has_latent(x, y)

            if direct_xy and latent_xy and not direct_yx:
                true_smcm.loc[x_name, y_name] = 4
                true_smcm.loc[y_name, x_name] = 5
            elif direct_yx and latent_xy and not direct_xy:
                true_smcm.loc[x_name, y_name] = 5
                true_smcm.loc[y_name, x_name] = 4
            elif direct_xy and not direct_yx:
                true_smcm.loc[x_name, y_name] = 2
                true_smcm.loc[y_name, x_name] = 3
            elif direct_yx and not direct_xy:
                true_smcm.loc[x_name, y_name] = 3
                true_smcm.loc[y_name, x_name] = 2
            elif latent_xy:
                true_smcm.loc[x_name, y_name] = 2
                true_smcm.loc[y_name, x_name] = 2

    return pred_smcm, true_smcm, truth_df, all_labels, A, latent_set, labels, latent_nodes

def evaluate_local(pred_pag: pd.DataFrame, true_pag: pd.DataFrame, target: str) -> dict:
    if target not in pred_pag.index or target not in true_pag.index:
        raise ValueError(f"Target '{target}' not found in PAG nodes.")

    pred_target_row = pred_pag.loc[target, :] != 0
    pred_target_col = pred_pag.loc[:, target] != 0
    pred_adjacency = pred_target_row | pred_target_col

    true_target_row = true_pag.loc[target, :] != 0
    true_target_col = true_pag.loc[:, target] != 0
    true_adjacency = true_target_row | true_target_col

    if pred_adjacency.sum() == 0 and true_adjacency.sum() == 0:
        return {
            "Adjacency": {"Precision": 1.0, "Recall": 1.0, "F1 Score": 1.0},
            "Mark": {"Precision": 1.0, "Recall": 1.0, "F1 Score": 1.0},
        }

    if (pred_adjacency.sum() == 0 and true_adjacency.sum() != 0) or (
        pred_adjacency.sum() != 0 and true_adjacency.sum() == 0
    ):
        return {
            "Adjacency": {"Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0},
            "Mark": {"Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0},
        }

    adj_tp = int((pred_adjacency & true_adjacency).sum())
    adj_fp = int((pred_adjacency & (~true_adjacency)).sum())
    adj_fn = int(((~pred_adjacency) & true_adjacency).sum())

    adj_precision = adj_tp / (adj_tp + adj_fp) if (adj_tp + adj_fp) > 0 else 0.0
    adj_recall = adj_tp / (adj_tp + adj_fn) if (adj_tp + adj_fn) > 0 else 0.0
    adj_f1 = 2 * (adj_precision * adj_recall) / (adj_precision + adj_recall) if (adj_precision + adj_recall) > 0 else 0.0

    evaluated_adjacency = pred_adjacency | true_adjacency
    adjacent_nodes = [
        node for node in evaluated_adjacency[evaluated_adjacency].index
        if node != target
    ]

    if len(adjacent_nodes) == 0:
        mark_tp = mark_fp = mark_fn = 0
        mark_precision = mark_recall = mark_f1 = 0.0
    else:
        mark_tp = mark_fp = mark_fn = 0

        def _predicted_endpoint_marks(code_xy: int, code_yx: int):
            if code_xy == 1 and code_yx == 1:
                return "circle", "circle"
            if code_xy == 2 and code_yx == 3:
                return "tail", "arrow"
            if code_xy == 3 and code_yx == 2:
                return "arrow", "tail"
            if code_xy == 2 and code_yx == 2:
                return "arrow", "arrow"
            if code_xy == 4 and code_yx == 5:
                return "circle", "arrow"
            if code_xy == 5 and code_yx == 4:
                return "arrow", "circle"
            return None, None

        def _allowed_true_endpoint_marks(code_xy: int, code_yx: int):
            if code_xy == 4 and code_yx == 5:
                return {"circle", "tail"}, {"arrow"}
            if code_xy == 5 and code_yx == 4:
                return {"arrow"}, {"circle", "tail"}
            first, second = _predicted_endpoint_marks(code_xy, code_yx)
            return ({first} if first else set(), {second} if second else set())

        def _score_mark(pred_mark: str | None, allowed: set[str]):
            if pred_mark is not None and allowed:
                if pred_mark in allowed:
                    return 1, 0, 0
                return 0, 1, 1
            if pred_mark is not None and not allowed:
                return 0, 1, 0
            if pred_mark is None and allowed:
                return 0, 0, 1
            return 0, 0, 0
        for node in adjacent_nodes:
            pred_code_1 = int(pred_pag.loc[target, node])
            pred_code_2 = int(pred_pag.loc[node, target])
            code_xy = int(true_pag.loc[target, node])
            code_yx = int(true_pag.loc[node, target])
            pred_mark_1, pred_mark_2 = _predicted_endpoint_marks(pred_code_1, pred_code_2)
            allowed_1, allowed_2 = _allowed_true_endpoint_marks(code_xy, code_yx)

            tp, fp, fn = _score_mark(pred_mark_1, allowed_1)
            mark_tp += tp
            mark_fp += fp
            mark_fn += fn

            tp, fp, fn = _score_mark(pred_mark_2, allowed_2)
            mark_tp += tp
            mark_fp += fp
            mark_fn += fn

        mark_precision = mark_tp / (mark_tp + mark_fp) if (mark_tp + mark_fp) > 0 else 0.0
        mark_recall = mark_tp / (mark_tp + mark_fn) if (mark_tp + mark_fn) > 0 else 0.0
        mark_f1 = 2 * (mark_precision * mark_recall) / (mark_precision + mark_recall) if (mark_precision + mark_recall) > 0 else 0.0

    return {
        "Adjacency": {
            "Precision": adj_precision,
            "Recall": adj_recall,
            "F1 Score": adj_f1,
        },
        "Mark": {
            "Precision": mark_precision,
            "Recall": mark_recall,
            "F1 Score": mark_f1,
        },
    }


