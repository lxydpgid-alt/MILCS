"""
For storing separation sets.
"""
from collections import defaultdict
from .mixgraph import Node
from typing import Optional
import warnings

class Separation_Set:

    def __init__(self, node_set: set[Node]):
        """
        Initialize the Separation_Set with a set of nodes.
        """
        if not node_set:
            warnings.warn("node_set must be a non-empty set of nodes.", UserWarning)

        self.node_set = node_set
        # key: (node1, node2)
        # val: set[frozenset[Node]]
        # Keep all discovered separation sets for each node pair, with duplicates removed.
        self.sepset = defaultdict(set)

    def _ensure_order(self, node1: Node, node2: Node) -> tuple[Node, Node]:
        """
        Ensure consistent ordering of node pairs.
        """
        return (node1, node2) if node1.index < node2.index else (node2, node1)

    def _add(self, node1: Node, node2: Node, sepset: set[Node]):
        """
        Add a separation set for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        self.sepset[(node1, node2)].add(frozenset(sepset))

    def has_sepset(self, node1: Node, node2: Node) -> bool:
        """
        Check if a separation set exists for the given pair of nodes.
        """
        node1, node2 = self._ensure_order(node1, node2)
        return (node1, node2) in self.sepset and len(self.sepset[(node1, node2)]) > 0

    def get_sepset(self, node1: Node, node2: Node):
        node1, node2 = self._ensure_order(node1, node2)
        if (node1, node2) in self.sepset:
            return [set(s) for s in self.sepset[(node1, node2)]]
        else:
            return None

    def is_in_sepset(self, target: Node, node1: Node, node2: Node) -> Optional[bool]:
        """
        Check if a target node is in the separation set of the given pair of nodes.
        Returns None if no separation set exists for the pair.

        """
        node1, node2 = self._ensure_order(node1, node2)
        if not self.has_sepset(node1, node2):
            return None
        # Return True as long as target appears in any candidate separation set.
        return any(target in cand for cand in self.sepset[(node1, node2)])

    def __repr__(self):
        """
        Return a string representation of the separation set dictionary.
        """
        result = []
        for (node1, node2), sepsets in self.sepset.items():
            formatted = [sorted([n.name for n in s]) for s in sepsets]
            result.append(f"{node1} - {node2}: {formatted}")
        return "\n".join(result)
