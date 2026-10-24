import os
import argparse
import json
import pandas as pd
import time
import re
from utils.PMB_learn import PMB_Learner
from Graph.mixgraph import MixGraph,Mark,Node,Edge
from Graph.G_orient import G_orient
from utils.print_g import print_g
from utils.evaluate import (
    build_eval_matrices,
    evaluate_local,
)

# Step 1: Load interventional sample tables
def load_data(folder: str):
    folder = os.path.expanduser(folder)
    pattern = re.compile(r"^data_\d+\.csv$")
    files = sorted(
        f for f in os.listdir(folder)
        if pattern.match(f)
    )
    dags = {}
    skipped = []
    for f in files:
        path = os.path.join(folder, f)
        # Read continuous data as sample tables.
        # Guard against transient empty-file windows when data generator is writing.
        last_err = None
        for k in range(3):
            try:
                df = pd.read_csv(path)
                if df.shape[1] == 0:
                    raise pd.errors.EmptyDataError("No columns parsed from CSV.")
                dags[f] = df
                break
            except (pd.errors.EmptyDataError, ValueError) as e:
                last_err = e
                time.sleep(0.2 * (k + 1))
        else:
            skipped.append((f, str(last_err) if last_err else "LoadError"))
    if skipped:
        print(f"Warning: Skipped unreadable interventional files: {skipped}")
    if not dags:
        raise ValueError(
            f"No valid interventional CSV loaded from {folder}. "
            f"Candidates={files}, skipped={skipped}"
        )
    print(f" Loaded {len(dags)} interventional DAGs")
    return dags
# Step 2: Local-learning entry point for continuous data: learn the skeleton first, then orient locally.
def learn_local(learner: PMB_Learner, target: str):
    learner.orientation_records = {} # Clear leftover records from the previous round so this round starts cleanly.
    learner._record_step = 0 # Previously used for debugging.
    skeleton = learner.learn_skeleton(target)
    part_directed_graph = learner.orient_collider(skeleton)
    local_pag = learner.orient_rules(part_directed_graph)
    return local_pag
# For the same target, learn a local PAG in each environment and accumulate each environment's sepset.
def collect_local_pags(
    target_name: str,
    dags: dict,
    env_learners: dict,
    global_sepsets: dict,
) -> list:
    local_pags = []
    for name, dag_df in dags.items():
        if target_name not in dag_df.columns:
            continue
        # Retrieve the PMB_Learner object for this environment from env_learners.
        learner = env_learners[name]
        local_pag = learn_local(learner, target_name)
        local_pags.append(local_pag)
        if name not in global_sepsets:
            # Each round returns a separation set.
            global_sepsets[name] = learner.sepsets
        else:
            # Merge the separation set returned in this round.
            global_sepsets[name] = merge_sepsets(global_sepsets[name], learner.sepsets)#old, new
    return local_pags
# One full round for the current target: local learning, global graph merge, edge pruning, then global orientation.
def process_target(
    target_name: str,
    G: MixGraph,
    dags: dict,
    env_learners: dict,
    global_sepsets: dict,
    orienter,
    pending_removed_edges: set,
) -> tuple[MixGraph, list]:
    """
    A uniform point-by-point processing method is used for
    the initial target round and all subsequent incremental rounds.
    """
    # Learn the local PAG for the current target node.
    local_pags = collect_local_pags(
        target_name=target_name,
        dags=dags,
        env_learners=env_learners,
        global_sepsets=global_sepsets,
    )
    if not local_pags:
        return G, local_pags
    G = merge_pags(
        local_pags=local_pags,
        target_name=target_name,
        G=G,
    )
    G = prune_edges(G, local_pags, pending_removed_edges)
    G = orienter.orient_rules(G, global_sepsets)
    return G, local_pags
# Add a missing edge or refine an existing one; only handle edges adjacent to the current node of this round.
def merge_pags(local_pags, target_name, G=None):
    def edge_key(a, b):
        return tuple(sorted((a.name, b.name)))
    def find_node(pag, name):
        return next((n for n in pag.node_list if n.name == name), None)
    def merge_edge(G, uG, vG, edge_L):
        # Skip this edge if it is already in removed_edges.
        if edge_key(uG, vG) in removed_edges:
            return
        # local_pag.get_Edge(u, v) is already aligned to the u->v viewpoint.
        l_u, l_v = edge_L.lmark, edge_L.rmark
        # Add the edge directly if it does not exist yet.
        if not G.has_edge(uG, vG):
            G.add_Edge(uG, vG, Edge(uG, l_u, l_v, vG))
            return
        # Conflicting definite marks become circles and stay unresolved on later merges.
        edge_G = G.get_Edge(uG, vG)  # The returned Edge object aligns its left/right marks to the input node order.
        g_u, g_v = edge_G.lmark, edge_G.rmark

        conflicts = G.graph.setdefault("merge_conflicts", set())

        def merge_mark(node, current, incoming):
            endpoint = (edge_key(uG, vG), node.name)
            if endpoint in conflicts:
                return Mark.CIRCLE
            if current == Mark.CIRCLE:
                return incoming
            if incoming == Mark.CIRCLE or current == incoming:
                return current
            conflicts.add(endpoint)
            return Mark.CIRCLE

        new_u = merge_mark(uG, g_u, l_u)
        new_v = merge_mark(vG, g_v, l_v)

        if (new_u, new_v) != (g_u, g_v):
            G.update_Edge(uG, new_u, new_v, vG)
    # Determine which node is being processed in the current round.
    current_name = target_name
    # First round: initialize the graph.
    if G is None:
        all_nodes = {}
        for pag in local_pags:
            for n in pag.node_list:
                all_nodes.setdefault(n.name, Node(n.name, n.index))
        G = MixGraph(incoming_graph_data=list(all_nodes.values()))
        g_nodes = all_nodes
    else:
        g_nodes = {n.name: n for n in G.node_list}
    # First add any nodes that appear in this round's local PAG but are not yet in G.
    for pag in local_pags:
        Z = find_node(pag, current_name)
        if Z is None:
            continue
        for n in pag.node_list:
            if n.name not in g_nodes:
                g_nodes[n.name] = Node(n.name, n.index)
                G.add_Node(g_nodes[n.name])

    # Only handle edges adjacent to the current node of this round.
    for pag in local_pags:
        Z = find_node(pag, current_name)
        if Z is None:
            continue
        ZG = g_nodes[Z.name]
        for X in pag.get_adj_nodes(Z):
            # Get the corresponding node object for neighbor X in G.
            XG = g_nodes[X.name]
            # Align by direction.
            edge_L = pag.get_Edge(Z, X)
            merge_edge(G, ZG, XG, edge_L)

    return G
# Collect induced edges first, then delete them in one batch.
def prune_edges(G: MixGraph, local_pags: list, pending_removed_edges: set):
    # Clear the pending deletion set first so it only records edges newly found in this round.
    pending_removed_edges.clear()

    # Only iterate over edges that currently exist in G; no need to enumerate all node pairs.
    for XG, YG in list(G.edges()):
        key = tuple(sorted((XG.name, YG.name)))

        # Skip edges that have already been permanently removed.
        if key in removed_edges:
            continue

        X_name, Y_name = XG.name, YG.name

        # Search each local PAG for evidence that this edge should be removed.
        for local_pag in local_pags:
            # Build a name-to-node mapping for lookup convenience.
            name_to_node = {n.name: n for n in local_pag.node_list}

            # If this local PAG does not even contain X or Y, it cannot provide evidence.
            if X_name not in name_to_node or Y_name not in name_to_node:
                continue

            X = name_to_node[X_name]
            Y = name_to_node[Y_name]

            # Condition 1: X and Y must be non-adjacent in the local PAG.
            if local_pag.has_edge(X, Y):
                continue

            # Check whether a node has at least one incoming arrowhead in the local PAG.
            # Since get_Edge(V, U) is aligned to the V -> U viewpoint, the mark at V is lmark.
            def has_arrowhead(V):
                for U in local_pag.get_adj_nodes(V):
                    if local_pag.get_Edge(V, U).lmark == Mark.ARROW:
                        return True
                return False

            # Condition 2: both X and Y must have appeared with an arrowhead.
            if has_arrowhead(X) and has_arrowhead(Y):
                pending_removed_edges.add(key)
                break

    # Apply deletions in one batch to avoid modifying G during iteration.
    for x_name, y_name in pending_removed_edges:
        XG = next((n for n in G.node_list if n.name == x_name), None)
        YG = next((n for n in G.node_list if n.name == y_name), None)

        if XG is None or YG is None:
            continue

        if G.has_edge(XG, YG):
            G.remove_Edge(XG, YG)

        # Record it in the permanent blacklist so merge_pags will not add it back later.
        removed_edges.add((x_name, y_name))

    # The pending deletion set for this round has been fully processed; clear it.
    pending_removed_edges.clear()
    return G
# Encapsulate the stopping rules for the main loop and return the stop reason when triggered.
def stop(
    G: MixGraph,
    target_name: str,
    done_count: int,
    max_observed_nodes: int | None,
    max_processed_nodes: int | None = 15,
) -> str | None:
    """
    Minimal stopping logic for the global expansion loop:
    1. Stop when there are no unresolved circle endpoints around the target.
    2. Stop as a safeguard when done_count exceeds the total number of observed nodes.
    3. Stop when done_count exceeds the user-defined processed-node cap.
    """
    target = None
    for n in G.node_list:
        if n.name == target_name:
            target = n
            break
    if target is None:
        raise ValueError(f"Target node {target_name} is not in the graph.")

    unresolved_circle = False
    for nb in G.get_adj_nodes(target):
        edge = G.get_Edge(target, nb)
        if edge.lmark == Mark.CIRCLE or edge.rmark == Mark.CIRCLE:
            unresolved_circle = True
            break

    if not unresolved_circle:
        return f"target {target_name} has no unresolved circle endpoints"

    if max_observed_nodes is not None and done_count > max_observed_nodes:
        return f"processed-node guard hit at target {target_name}"

    if max_processed_nodes is not None and done_count > max_processed_nodes:
        return f"processed-node cap {max_processed_nodes} hit at target {target_name}"

    return None
# Extend the waitlist using neighbors of the current target in the global graph.
def update_waitlist(
    local_pags: list,
    target_name: str,
    waitlist: list,
    donelist: set,
    G: MixGraph | None = None,
    policy: str = "all_local",
):

    if policy not in {"all_local", "circle_circle", "at_least_one_circle"}:
        raise ValueError(f"Unknown waitlist policy: {policy}")
    if policy != "all_local" and G is None:
        raise ValueError(f"G is required by the {policy} waitlist policy")

    seen = set(waitlist)
    global_nodes = {node.name: node for node in G.node_list} if G is not None else {}
    global_target = global_nodes.get(target_name)
    for local_pag in local_pags:
        if target_name not in {n.name for n in local_pag.node_list}:
            continue
        for node in local_pag.node_list:
            if node.name == target_name or node.name in LATENT_VARS:
                continue
            if node.name in donelist or node.name in seen:
                continue
            if policy != "all_local":
                global_node = global_nodes.get(node.name)
                if (
                    global_target is None
                    or global_node is None
                    or not G.has_edge(global_target, global_node)
                ):
                    continue
                edge = G.get_Edge(global_target, global_node)
                target_mark = edge.lmark
                node_mark = edge.rmark
                if policy == "circle_circle" and not (
                    target_mark == Mark.CIRCLE and node_mark == Mark.CIRCLE
                ):
                    continue
                if policy == "at_least_one_circle" and not (
                    target_mark == Mark.CIRCLE or node_mark == Mark.CIRCLE
                ):
                    continue
            waitlist.append(node.name)
            seen.add(node.name)
# Accumulate sepsets across rounds within the same environment by union.
def merge_sepsets(old_sepset, new_sepset):
    if old_sepset is None:
        return new_sepset
    old_sepset.node_set |= new_sepset.node_set  # Union node sets.
    for key, bucket in new_sepset.sepset.items():
        old_sepset.sepset[key].update(bucket)# Union separation sets.
    return old_sepset


if __name__ == "__main__":
    example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example")
    # parser is used to parse command-line arguments.
    parser = argparse.ArgumentParser(description="continuous local/global learning runner")
    parser.add_argument("--target", default="V63", help="Target node name, e.g. V20")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for CI tests.",
    )
    parser.add_argument(
        "--truth-matrix",
        default=os.path.join(example_dir, "truth_matrix.csv"),
        help="Truth 0/1 matrix CSV path",
    )
    parser.add_argument(
        "--intervention-dir",
        default=example_dir,
        help="Directory containing data_i.csv",
    )
    parser.add_argument(
        "--ci-type",
        default="Fisher_Z",
        choices=["Fisher_Z", "KCI", "RCIT", "FastKCI", "Poly_Z", "G_sq"],
        help="CI mode for continuous or discrete data.",
    )
    parser.add_argument(
        "--max-processed-nodes",
        type=int,
        default=None,
        help=(
            "Stop if processed nodes exceed this cap. By default, networks "
            "with more than 50 nodes use floor(20%% of all nodes), matching "
            "the paper's experimental protocol."
        ),
    )
    parser.add_argument(
        "--waitlist-policy",
        choices=["all_local", "circle_circle", "at_least_one_circle"],
        default="all_local",
        help=(
            "all_local keeps the original expansion; at_least_one_circle "
            "queues nodes whose edge to the current target has a circle."
        ),
    )
    args = parser.parse_args()
# ---- Runtime configuration and global state ----
    # Record the start time of the full pipeline for total runtime statistics.
    t_pipeline_start = time.perf_counter()
    # Current conditional independence test type.
    CI_TEST_TYPE = args.ci_type
    # Read the target node and truth-matrix path from command-line arguments.
    TARGET_NODE = args.target
    # Assume latent variable names are L1 to L100, used only when filtering observed nodes.
    LATENT_VARS = {f"L{i}" for i in range(1, 101)}
    TRUTH_MATRIX_PATH = os.path.expanduser(args.truth_matrix)
    removed_edges = set()  # Record edges that have been permanently removed.
    pending_removed_edges = set()  # Record edges newly found this round and pending batch removal.
    global_sepsets = {}
    orienter = G_orient()
    # Initialize the global graph.
    G = None
    DoneList = set()
    WaitList = [TARGET_NODE]
    # ---- Load data from each environment ----
    dags = load_data(folder=args.intervention_dir)
    observed_node_names = set()
    for dag_df in dags.values():
        observed_node_names.update(
            col for col in dag_df.columns if col not in LATENT_VARS
        )
    max_observed_nodes = len(observed_node_names) # Safeguard stopping limit.
    total_nodes = pd.read_csv(TRUTH_MATRIX_PATH, index_col=0).shape[0]
    max_processed_nodes = args.max_processed_nodes
    if max_processed_nodes is None and total_nodes > 50:
        max_processed_nodes = max(1, int(0.2 * total_nodes))
    print(
        f"Iteration guard: total_nodes={total_nodes}, "
        f"max_processed_nodes={max_processed_nodes}"
    )
    # Each environment maintains its own PMB_Learner so CI caches, sepsets, and statistics stay independent.
    env_learners = {}
    for name, dag_df in dags.items():
        learner_kwargs = dict(
            data=dag_df,
            ci_type=CI_TEST_TYPE,
            alpha=args.alpha,
            max_K=3,
        )
        # Create a PMB_Learner for this environment and store it in env_learners.
        env_learners[name] = PMB_Learner(**learner_kwargs)
    first_learner = next(iter(env_learners.values()))
    mb_learner = first_learner.mb_learner
    print(
        "MB learner: "
        f"{type(mb_learner).__name__}, "
        f"data_type={getattr(mb_learner, 'data_type', 'n/a')}, "
        f"max_k={getattr(mb_learner, 'max_k', 'n/a')}, "
        f"CI={CI_TEST_TYPE}"
    )
    print("===== Start =====")
    print(f"Target node: {TARGET_NODE}")
    print(f"alpha: {args.alpha}")
    print(f"WaitList policy: {args.waitlist_policy}")
# ---- Iteration ----
    # Each round pops one node from WaitList:
    # 1. Learn a local PAG for that node in every environment.
    # 2. Merge the local information into the global graph G.
    # 3. Collect and apply induced-edge deletions.
    # 4. Perform global orientation using accumulated sepsets.
    # 5. Update the next batch of nodes to process and check the stop rule.
    loop_exit_reason = "waitlist exhausted"
    while WaitList:
        Z = WaitList.pop(0)             # Pop the node to process in the current round.
        if Z in DoneList:           # Skip it if this target has already been processed.
            continue
        DoneList.add(Z)
        if G is None:               # This means it is the first round.
            print(" Building the initial global causal structure G0")
        else:
            print("------------------------------------")
            print(f" Processing node {Z}")
        # Uniformly process the current target: local learning -> merge into G -> prune edges -> global orientation.
        G, local_pags = process_target(
            target_name=Z,
            G=G,
            dags=dags,
            env_learners=env_learners,
            global_sepsets=global_sepsets,
            orienter=orienter,
            pending_removed_edges=pending_removed_edges,
        )
        if G is None:
            continue
        update_waitlist(
            local_pags,
            Z,
            WaitList,
            DoneList,
            G=G,
            policy=args.waitlist_policy,
        )
        # stop() encapsulates the stopping logic for the current main loop.
        stop_reason = stop(
            G=G,
            target_name=TARGET_NODE,
            done_count=len(DoneList),
            max_observed_nodes=max_observed_nodes,
            max_processed_nodes=max_processed_nodes,
        )
        if stop_reason is not None:
            loop_exit_reason = stop_reason
            print(f" Stop: {stop_reason}")
            break
    if G is None:
        raise RuntimeError("No global graph was constructed for the target node.")

    processed_observed_nodes = DoneList & observed_node_names
    unprocessed_observed_nodes = observed_node_names - DoneList
    traversal_stats = {
        "processed_observed": len(processed_observed_nodes),
        "total_observed": len(observed_node_names),
        "coverage": len(processed_observed_nodes) / len(observed_node_names),
        "traversed_all": not unprocessed_observed_nodes,
        "exit_reason": loop_exit_reason,
        "waitlist_policy": args.waitlist_policy,
        "unprocessed_nodes": sorted(unprocessed_observed_nodes),
    }
    print("TRAVERSAL_STATS " + json.dumps(traversal_stats, sort_keys=True))

    print("\n Final learned G (PAG edges):")
    print_g(G)

# ---- Result evaluation and output ----
    # After the main loop ends, evaluate only the target's local PAG.
    pred_smcm, true_smcm, _, _, _, _, labels, _ = build_eval_matrices(
        G=G,
        truth_matrix_path=TRUTH_MATRIX_PATH,
        latent_vars=LATENT_VARS,
    )
    if TARGET_NODE not in labels:
        raise ValueError(f"Target {TARGET_NODE} not found in comparable observed labels.")
    local_metrics = evaluate_local(pred_smcm, true_smcm, TARGET_NODE)

    # Aggregate CI-test counts across all environments and compute the total pipeline runtime.
    ci_total = sum(learner.get_ci_test_number() for learner in env_learners.values())
    runtime_total = time.perf_counter() - t_pipeline_start

    print("\n Local target metrics (evaluate_local style):")
    print(
        f" Adjacency Precision, Recall, F1: "
        f"{local_metrics['Adjacency']['Precision']:.4f}, "
        f"{local_metrics['Adjacency']['Recall']:.4f}, "
        f"{local_metrics['Adjacency']['F1 Score']:.4f}"
    )
    print(
        f"Mark Precision, Recall, F1: "
        f"{local_metrics['Mark']['Precision']:.4f}, "
        f"{local_metrics['Mark']['Recall']:.4f}, "
        f"{local_metrics['Mark']['F1 Score']:.4f}"
    )
    print(f"  - CI tests/Runtime: {ci_total}, {runtime_total:.3f}s")
