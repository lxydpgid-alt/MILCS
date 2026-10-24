"""Print mixed graph edges using their existing text representation."""

from Graph.mixgraph import MixGraph


def print_g(G: MixGraph) -> None:
    """Print each edge in G once using Edge.__repr__."""
    for _, _, data in G.edges(data=True):
        print(data["edge"])
