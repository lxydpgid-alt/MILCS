from typing import Optional, Dict
import logging
from Graph.mixgraph import MixGraph, Mark, Node
from Graph.sepset import Separation_Set

logger = logging.getLogger(__name__)

class G_orient:
    def __init__(self):
        # key: (min(node_name), max(node_name))
        # value: list[record]
        self.orientation_records = {}
        self._record_step = 0

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
        """
        Update an edge and store orientation record if edge marks changed.
        Returns True if changed, otherwise False.
        """
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
        record = {
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
        self.orientation_records.setdefault(key, []).append(record)
        self._record_step += 1
        return True

    def _resolve_nodes_in_env(
        self,
        ss: Separation_Set,
        x: Node,
        y: Node,
        target: Node,
    ):
        name_to_node = {n.name: n for n in ss.node_set}
        x_env = name_to_node.get(x.name)
        y_env = name_to_node.get(y.name)
        if x_env is None or y_env is None:
            return None
        t_env = name_to_node.get(target.name)
        if t_env is None:
            return None
        return x_env, y_env, t_env

    def _any_in_sepset(self, local_sepsets: Dict[str, Separation_Set], target: Node, x: Node, y: Node) -> bool:
        for ss in local_sepsets.values():
            resolved = self._resolve_nodes_in_env(ss, x, y, target)
            if resolved is None:
                continue
            x_env, y_env, t_env = resolved
            if ss.has_sepset(x_env, y_env) and ss.is_in_sepset(
                target=t_env, node1=x_env, node2=y_env
            ):
                return True
        return False


    def Rule_1(
        self,
        pag: MixGraph,
        local_sepsets: Dict[str, Separation_Set],
        update_flag: bool,
    ):
        """
        If b o-* r, and a *-> b, a and r are not adjacent, and b in SepSet(a,r),
        then orient b o-* r as b -> r.
        Here, SepSet(a,r) / membership are checked in LOCAL sepsets, any env satisfies => orient.
        """
        for b, r in pag.get_circ_star_Edge():
            for a in pag.get_into_nodes(b):
                if a == b or a == r:
                    continue
                if pag.has_edge(a, r):
                    continue
                if self._any_in_sepset(local_sepsets, b, a, r):
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
                    logger.info(f"[R1] {b.name} o-* {r.name} ==> {b.name} -> {r.name}")
                    break
            if update_flag:
                break

        return pag, update_flag

    def orient_rules(self, G: MixGraph, local_sepsets: Dict[str, Separation_Set]) -> MixGraph:
        """
        Apply rules iteratively until convergence.
        Uses LOCAL sepsets only (any env supports).
        """
        update_flag = True
        while update_flag:
            update_flag = False
            G, update_flag = self.Rule_1(G, local_sepsets, update_flag)

        return G
