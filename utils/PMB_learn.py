import logging
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import pandas as pd

from Graph.mixgraph import MixGraph, Mark, Node
from Graph.sepset import Separation_Set
from .CI_test import CausalLearnCITest
from .mb import FastIAMB_learn, discrete_MB_learn, gaussian_MB_learn


"""
NOTE: The 'Node' only be used in  the Graph learning process, we don't use it in the other process, e.g., MB learning.
"""
logger = logging.getLogger("PMB_Learner")
logger.setLevel(logging.INFO)


class PMB_Learner:

    def __init__(self, data, alpha = 0.05, ci_type = "Fisher_Z", **kwargs):

        self.ci_type = ci_type

        observed_data = data

        if ci_type == "Fisher_Z":
            # New Fisher-Z implementation lives in CI_test.py and accepts only DataFrame.
            # Keep PMB_Learner compatible with ndarray callers by wrapping once here.
            ci_data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(
                data, columns=[i for i in range(data.shape[1])]
            )
            ci_kwargs = {}
            if "Max_time" in kwargs:
                ci_kwargs["Max_time"] = kwargs["Max_time"]
            elif "max_time" in kwargs:
                ci_kwargs["Max_time"] = kwargs["max_time"]
            if "use_cache" in kwargs:
                ci_kwargs["use_cache"] = kwargs["use_cache"]

            self.ci_test = CausalLearnCITest(
                ci_data,
                method="fisherz",
                alpha=alpha,
                **ci_kwargs,
            )
            # Compatibility attributes expected elsewhere in the old CI chain.
        elif ci_type in {"G_sq", "KCI", "RCIT", "FastKCI", "Poly_Z"}:
            method = {
                "G_sq": "gsq",
                "KCI": "kci",
                "RCIT": "rcit",
                "FastKCI": "fastkci",
                "Poly_Z": "polyz",
            }[ci_type]
            self.ci_test = CausalLearnCITest(
                data if isinstance(data, pd.DataFrame) else pd.DataFrame(data),
                method=method,
                alpha=alpha,
                use_cache=kwargs.get("use_cache", True),
            )
        if ci_type == "Fisher_Z":
            self.mb_learner = gaussian_MB_learn(
                data=observed_data,
                alpha=alpha,
                **kwargs,
            )
        elif ci_type == "G_sq":
            self.mb_learner = FastIAMB_learn(
                data=observed_data,
                ci_test=self.ci_test,
                alpha=alpha,
                max_k=kwargs.get("max_K", 3),
                data_type="discrete",
            )
        elif ci_type in {"KCI", "RCIT", "FastKCI", "Poly_Z"}:
            self.mb_learner = discrete_MB_learn(
                data=observed_data,
                ci_test=self.ci_test,
                max_conditioning_size=kwargs.get("max_K", 15),
            )
        else:
            raise ValueError(f"Unsupported ci_type: {ci_type}")
        self.max_K = kwargs.get("max_K", 15)  # Maximum size of the separation set, default is 100
        self._init_nodes(observed_data)
        self.orientation_records: Dict[tuple, List[Dict]] = {}
        self._record_step = 0


    def get_mb(self, target_node: Node) -> set[Node]:

        mb_names = self.mb_learner(target_node.name)
        mb_nodes = {self.Nodes_dict[name] for name in mb_names}
        return mb_nodes

    def get_ci_test_number(self) -> int:
        """
        Get the number of CI tests performed.
        """
        return self.ci_test.get_ci_num()  # including the number of CI tests performed in the MB learning process and skeleton learning process

    def _edge_key(self, node1: Node, node2: Node):
        return tuple(sorted([node1.name, node2.name]))

    def _mark_to_str(self, mark):
        if mark is None:
            return None
        return str(mark)

    def _record_orient(
        self,
        pag: MixGraph,
        node1: Node,
        node2: Node,
        lmark: Mark,
        rmark: Mark,
        rule: str,
        condition: Optional[Dict] = None,
    ) -> bool:
        old_edge = pag.get_Edge(node1, node2)
        if old_edge is None:
            return False

        old_lmark = old_edge.lmark
        old_rmark = old_edge.rmark

        pag.update_Edge(node1=node1, lmark=lmark, rmark=rmark, node2=node2)
        new_edge = pag.get_Edge(node1, node2)
        if new_edge is None:
            return False

        changed = (old_lmark != new_edge.lmark) or (old_rmark != new_edge.rmark)
        if not changed:
            return False

        key = self._edge_key(node1, node2)
        self.orientation_records.setdefault(key, []).append(
            {
                "step": self._record_step,
                "rule": rule,
                "edge": (node1.name, node2.name),
                "original": {
                    "lmark": self._mark_to_str(old_lmark),
                    "rmark": self._mark_to_str(old_rmark),
                },
                "new": {
                    "lmark": self._mark_to_str(new_edge.lmark),
                    "rmark": self._mark_to_str(new_edge.rmark),
                },
                "condition": condition or {},
            }
        )
        self._record_step += 1
        return True


    def _init_nodes(self, data: pd.DataFrame):

        if isinstance(data, pd.DataFrame):
            self.Nodes_list = [Node(node_name, index) for index, node_name in enumerate(data.columns)]
            self.Nodes_dict = {node.name: node for node in self.Nodes_list} # PMB_Learner.Nodes_dict[node_name] = Node, node_name is str
        else:
            raise TypeError("Data must be a pandas DataFrame.")

    def learn_skeleton_base_data(self, sub_nodes: list[Node]) -> Tuple[MixGraph, Separation_Set]:
        """
        Learn the skeleton based on a provided subset of nodes.
        This implementation is aligned with the provided learn_skeleton_base_data logic.
        It returns a pure skeleton; local orientation is handled by the caller.
        """
        sub_graph = MixGraph(incoming_graph_data=sub_nodes)
        logger.info(f'Initial graph information: {sub_graph}')
        sub_graph._init_complete_graph()
        sepsets = Separation_Set(set(sub_nodes))

        sep_size = 0
        while sep_size <= self.max_K and (sub_graph.max_degree() - 1 >= sep_size):
            for x_node in list(sub_graph.node_list):
                adj_x_nodes = set(sub_graph.get_adj_nodes(x_node))
                for y_node in list(adj_x_nodes):
                    if not sub_graph.has_edge(x_node, y_node):
                        continue
                    adj_node_x_noy = adj_x_nodes - {y_node}
                    adj_node_x_noy = set(node.name for node in adj_node_x_noy)
                    if len(adj_node_x_noy) < sep_size:
                        continue
                    logger.info(f'Checking edge {x_node} -- {y_node} for sep_size={sep_size}')
                    for sep in combinations(sorted(adj_node_x_noy), sep_size):
                        if self.ci_test(x_node.name, y_node.name, list(sep))[0]:
                            sep_nodes = {self.Nodes_dict[name] for name in sep}
                            sepsets._add(x_node, y_node, sep_nodes)
                            sub_graph.remove_Edge(x_node, y_node)
                            break
            sep_size += 1

        self.sepsets = sepsets
        return sub_graph, sepsets

    def learn_skeleton(self, target: str) -> MixGraph:
        """
        Learn a pure local skeleton on MB(target) U {target}.
        The caller is responsible for collider/orientation rules.
        """
        target_node = self.Nodes_dict[target]
        sub_nodes = sorted(self.get_mb(target_node) | {target_node}, key=lambda n: n.index)
        sub_graph, _ = self.learn_skeleton_base_data(sub_nodes)
        return sub_graph


    def orient_collider(self, undirected_graph: MixGraph) -> MixGraph:


        Cand_triplets = undirected_graph.find_unique_triplets()

        for (z, y, x) in Cand_triplets:

            if self.sepsets.has_sepset(x, z): # in local causal discovery, we check the existence of sepset of x and z
                if not self.sepsets.is_in_sepset(target=y, node1=x, node2=z):
                    self._record_orient(
                        pag=undirected_graph,
                        node1=x,
                        node2=y,
                        lmark=None,
                        rmark=Mark.ARROW,
                        rule="Collider",
                        condition={"reason": f"{y.name} not in SepSet({x.name}, {z.name})"},
                    )
                    self._record_orient(
                        pag=undirected_graph,
                        node1=z,
                        node2=y,
                        lmark=None,
                        rmark=Mark.ARROW,
                        rule="Collider",
                        condition={"reason": f"{y.name} not in SepSet({x.name}, {z.name})"},
                    )
                    logger.info(f"Orienting collider: {x} *-> {y} <-* {z}")

        return undirected_graph

    def orient_rules(self, pag: MixGraph) -> MixGraph:
        """
        Apply the orientation rules to the PAG.
        """
        update_flag = True
        while update_flag:  # Continue until no changes are made
            update_flag = False
            pag, update_flag = self.Rule_1(pag, update_flag)
            pag, update_flag = self.Rule_2(pag, update_flag)
            pag, update_flag = self.Rule_3(pag, update_flag)


        return pag

    def Rule_1(self, pag: MixGraph, update_flag: bool):
        for b, r in pag.get_circ_star_Edge():

            for a in pag.get_into_nodes(b):
            # Guard 2: all nodes must be distinct
                if a == b or a == r:
                    continue
            # Guard 3 (THEORETICAL REQUIREMENT):
            # a and r must NOT be adjacent
                if pag.has_edge(a, r):
                    continue
            # Separation-set condition (FCI semantics)
                if self.sepsets.has_sepset(a, r) and \
                   self.sepsets.is_in_sepset(target=b, node1=a, node2=r):

                    changed = self._record_orient(
                        pag=pag,
                        node1=b,
                        node2=r,
                        lmark=Mark.TAIL,
                        rmark=Mark.ARROW,
                        rule="Rule_1",
                        condition={"reason": f"{b.name} in SepSet({a.name}, {r.name})"},
                    )
                    update_flag = update_flag or changed
                    logger.info(f"[R1] {b} o-* {r}  ==>  {b} -> {r}")
                    break   # important: stop after first valid a
        return pag, update_flag

    def Rule_2(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a -> b *-> r or a *-> b -> r, and a *-o r, then orient a *-o r as a *-> r.
        """
        for r, a in pag.get_circ_star_Edge():  # edges of the form a *-o r
            for b in pag.get_into_nodes(r):  # b *-> r
                if pag.has_directed_Edge(a, b) or \
                    (pag.has_into_Edge(a, b) and pag.has_out_Edge(b, r)):
                    changed = self._record_orient(
                        pag=pag,
                        node1=a,
                        node2=r,
                        lmark=None,
                        rmark=Mark.ARROW,
                        rule="Rule_2",
                        condition={"reason": f"{a.name}->{b.name}*->{r.name} or {a.name}*->{b.name}->{r.name}"},
                    )
                    update_flag = update_flag or changed
                    logger.info(f"Orienting Rule 2: {a} *-o {r} to {a} *-> {r}")
                    break
        return pag, update_flag

    def Rule_3(self, pag: MixGraph, update_flag: bool) -> MixGraph:
        """
        If a *-> b <-* r, a *-o t o-* r, a and r are not adjacent, and t *-o b,
        then orient t *-o b as t *-> b.
        """
        for b, t in pag.get_circ_star_Edge():  # Get edges of the form t *-o b
            Cand_ar = pag.get_into_nodes(b)   # a or r *-> b
            Cand_ar = {a for a in Cand_ar if pag.has_into_Edge(a, b)}  # Only consider a *-> b
            if len(Cand_ar) >= 2:
                for a, r in combinations(Cand_ar, 2):
                    if self.sepsets.has_sepset(a, r) and \
                        self.sepsets.is_in_sepset(target=t, node1=a, node2=r):  # otherwise t is a collider on <a, t, r>
                            changed = self._record_orient(
                                pag=pag,
                                node1=t,
                                node2=b,
                                lmark=None,
                                rmark=Mark.ARROW,
                                rule="Rule_3",
                                condition={"reason": f"{t.name} in SepSet({a.name}, {r.name})"},
                            )
                            update_flag = update_flag or changed
                            logger.info(f"Orienting Rule 3: {t} *-o {b} to {t} *-> {b}")
                            break
        return pag, update_flag


