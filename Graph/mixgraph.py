from itertools import combinations
import networkx as nx
from typing import Optional, Set, Tuple, Union, List
import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum
class Mark(Enum):


    """Endpoint encoding used by the graph representation."""
    TAIL = 3   # "-"
    ARROW = 2  # ">"
    CIRCLE = 1 # "o"
    NULL = 0   # " "

    def __eq__(self, other):
        if isinstance(other, Mark):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o", Mark.NULL: " "}[self]

    def __repr__(self):
        return f"Mark.{self.name}"


from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    name: Optional[str] = None
    index: Optional[int] = None

    def __repr__(self):
        return f"{self.name}"
    def __str__(self):
        return f"{self.name}"


class Edge:
    def __init__(self, start: Node, lmark: Mark, rmark: Mark, end: Node):  # start lmark-rmark end
        self.start = start
        self.end = end
        self.lmark = lmark  # The mark here can be Mark.TAIL, Mark.ARROW, or Mark.CIRCLE
        self.rmark = rmark
    def __repr__(self):
        rmark_symbol = {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o"}
        lmark_symbol = {Mark.TAIL: "-", Mark.ARROW: "<", Mark.CIRCLE: "o"}
        return f"{self.start}{lmark_symbol[self.lmark]}-{rmark_symbol[self.rmark]} {self.end}"
    def _invert(self):
        # Inverts the direction of the edge
        self.start, self.end = self.end, self.start
        self.lmark, self.rmark = self.rmark, self.lmark

    def copy(self):
        return Edge(self.start, self.lmark, self.rmark, self.end)


class MixGraph(nx.Graph):
    def __init__(self, incoming_graph_data: Optional[Union[pd.DataFrame, np.ndarray, list]] = None):
        """
        Initialize a Mix Graph based on networkx.Graph.
        If incoming_graph_data is provided, it will be used to create the graph.
        pd.DataFrame or np.ndarray means the adjacency matrix. if list, it means the node list.
        """
        super().__init__()
        self._cache_path = defaultdict(dict)
        self._cache_nodes = defaultdict(dict)
        if incoming_graph_data is not None:
            if isinstance(incoming_graph_data, pd.DataFrame):
                self.from_pandas_adjacency(incoming_graph_data)
            elif isinstance(incoming_graph_data, np.ndarray):
                self.from_numpy_array(incoming_graph_data)
            elif isinstance(incoming_graph_data, list):
                self.from_node_list(incoming_graph_data)
            else:
                raise TypeError("Invalid graph data type.")

    def clear_cache(self):
        """Clear the cache."""
        self._cache_path.clear()
        self._cache_nodes.clear()

    ####################
    ## Edge functions ##
    ####################

    def _ensure_order(self, node1: Node, node2: Node) -> tuple[Node, Node]:
        """
        Ensure node1 < node2 based on the integer mapping.
        """
        if node1.index > node2.index:
            return node2, node1
        return node1, node2


    def add_Edge(self, node1: Node, node2: Node, edge: Edge):
        """
        Add an edge to the graph.
        The parent function that for add different types of edges.
        """
        self.clear_cache()
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise TypeError("node1 and node2 must be of type Node.")
        if not isinstance(edge, Edge):
            raise TypeError("edge must be of type Edge.")
        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

        # If an edge exists, remove it first
        if self.has_edge(node1, node2):
            raise ValueError(f"Edge between {node1} and {node2} already exists. Use update_Edge() to update it.")

        # Ensure node1.index < node2.index
        n1, n2 = self._ensure_order(node1, node2)
        if (n1, n2) == (node2, node1):
            edge._invert()  # Invert the edge if the order is reversed

        self.add_edge(node1, node2, edge=edge)


    def remove_Edge(self, node1: Node, node2: Node):
        """
        Remove an edge between two nodes if it exists.
        """
        self.clear_cache()
        if not isinstance(node1, Node) or not isinstance(node2, Node):
            raise TypeError("node1 and node2 must be of type Node.")
        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before removing an edge.")

        if self.has_edge(node1, node2):
            self.remove_edge(node1, node2)
        else:
            raise ValueError(f"Edge between {node1} and {node2} does not exist.")


    def add_circ_Edge(self, node1: Node, node2: Node):
        """
        Set an Undirected edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-o node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        # Add the new edge with the specified marks
        edge = Edge(node1, Mark.CIRCLE, Mark.CIRCLE, node2)
        self.add_Edge(node1, node2, edge=edge)


    def add_directed_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes if it does not already exist. If the edge already exists, it will not be replaced.
        node1 -> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        edge = Edge(node1, Mark.TAIL, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)


    def add_bidirected_Edge(self, node1: Node, node2: Node):
        """
        Add a bidirected edge between two nodes if it does not already exist. If the edge already exists, it will not be replaced.
        node1 <-> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        edge = Edge(node1, Mark.ARROW, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)


    def add_circ_arrow_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-> node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()
        edge = Edge(node1, Mark.CIRCLE, Mark.ARROW, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)


    def add_circ_tail_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with CIRCLE marks. If the edge already exists, it will not be replaced.
        node1 o-- node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()

        edge = Edge(node1, Mark.CIRCLE, Mark.TAIL, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)


    def add_tail_Edge(self, node1: Node, node2: Node):
        """
        Add a directed edge between two nodes with TAIL marks. If the edge already exists, it will not be replaced.
        node1 --- node2
        Parameters:
        - node1: The first node.
        - node2: The second node.
        """
        self.clear_cache()

        edge = Edge(node1, Mark.TAIL, Mark.TAIL, node2)
        # Add the new edge with the specified marks
        self.add_Edge(node1, node2, edge=edge)


    def update_Edge(self, node1: Node, lmark: Mark, rmark: Mark, node2: Node):
        """
        Orient an edge between two nodes with specific marks. If the edge does not exist, it will not be created.

        Parameters:
        - node1: The first node.
        - node2: The second node.
        - lmark: The left mark.
        - rmark: The right mark.
        """
        self.clear_cache()

        if not self.has_node(node1) or not self.has_node(node2):
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

        if self.has_edge(node1, node2):

            # Ensure node1 < node2
            n1, n2 = self._ensure_order(node1, node2)
            if (n1, n2) == (node2, node1):
                lmark, rmark = rmark, lmark  # Swap marks if the order is reversed
                node1, node2 = node2, node1  # Swap nodes

            # Update the existing edge with new marks
            edge = self[node1][node2]['edge'].copy()  # Copy the existing edge object
            if lmark is not None:
                edge.lmark = lmark
            if rmark is not None:
                edge.rmark = rmark
            self.add_edge(node1, node2, edge=edge)  # Update the edge in the graph
        else:
            raise ValueError(f"Edge between {node1} and {node2} does not exist. Use add_xx_edge() to create it.")


    def get_Edge(self, start: Node, end: Node) -> Optional[Edge]:
        """
        Retrieve the edge object between two nodes if it exists.
        start --- end

        Parameters:
        - start: The starting node of the edge.
        - end: The ending node of the edge.

        Returns:
        - The Edge object (start--edge--end) if the edge exists, otherwise None.
        """

        if self.has_edge(start, end):
            if self[start][end]['edge'] is not None:
                edge = self[start][end]['edge'].copy()
            else:
                raise ValueError(f"get_Edge error, 'edge' between {start} and {end} does not exist.")
            if edge.start == end and edge.end == start:
                edge._invert()

            if edge.start != start or edge.end != end:
                raise ValueError(f"get_Edge error, Edge between {start} and {end} is not in the correct order.")
            return  edge # Invert the edge if the order is reversed
        return None

    def has_directed_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is a directed edge from the start node to the end node.
        start -> end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if a directed edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.TAIL and edge.rmark == Mark.ARROW)
        return False


    def has_into_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is an edge into the start node from the end node.
        start *-> end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if a into edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.rmark == Mark.ARROW)
        return False


    def has_out_Edge(self, start: Node, end: Node) -> bool:
        """
        Determine if there is an edge out of the start node to the end node.
        start --* end

        Parameters:
        - start: The source node.
        - end: The target node.

        Returns:
        - True if an out edge exists.
        - False otherwise.
        """
        edge = self.get_Edge(start, end)
        if edge is not None:
            return (edge.lmark == Mark.TAIL)
        return False


    def get_circ_star_Edge(self) -> List[tuple[Node, Node]]:
        """
        Get all edges formed by o-* in the graph.
        Returns:
            A list of tuples representing the edges formed by o-* in the graph.
            In the list, each tuple is of the form (start_node, end_node), start_node o-* end_node.
        """
        circ_star_edges = []
        for edge in self.edges(data=True):
            edge_obj = edge[2]['edge']
            if edge_obj.lmark == Mark.CIRCLE:  #  start o-* end
                circ_star_edges.append((edge_obj.start, edge_obj.end))
            if edge_obj.rmark == Mark.CIRCLE:  #  end o-* start
                circ_star_edges.append((edge_obj.end, edge_obj.start))

        return circ_star_edges


    def max_degree(self) -> int:
        """
        Get the maximum degree of the graph.
        """
        return max(dict(self.degree()).values(), default=0)


    ########################
    ## End Edge functions ##
    ########################

    ####################
    ## Path functions ##
    ####################


    ########################
    ## End Path functions ##
    ########################


    ###########################################
    ## Initializing Graphs in Different Ways ##
    ###########################################
    def from_numpy_array(self, adj_matrix: np.ndarray, node_list: List[Node] = None):
        """
        Create a Mixed Graph from a NumPy adjacency matrix.
        """
        if node_list is None:
            self.node_list = [Node(name=str(i), index=i) for i in range(adj_matrix.shape[0])]  # [Node(str, 0), Node(str, 1), ...]
        else:
            self.node_list = node_list

        self.add_nodes_from(self.node_list)
        for i, j in combinations(self.node_list, 2):
            mark_ij = adj_matrix[i.index, j.index]  # i markji --- markij j
            mark_ji = adj_matrix[j.index, i.index]


            if mark_ij == Mark.ARROW.value:
                if mark_ji == Mark.ARROW.value:    # i <-> j
                    self.add_bidirected_Edge(i, j)
                elif mark_ji == Mark.CIRCLE.value:  # i o-> j
                    self.add_circ_arrow_Edge(i, j)
                elif mark_ji == Mark.TAIL.value:     # i --> j
                    self.add_directed_Edge(i, j)
                else:
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.CIRCLE.value:
                if mark_ji == Mark.ARROW.value:    # i <-o j
                    self.add_circ_arrow_Edge(j, i)
                elif mark_ji == Mark.CIRCLE.value:  # i o-o j
                    self.add_circ_Edge(j, i)
                elif mark_ji == Mark.TAIL.value:  # i --o j
                    self.add_circ_tail_Edge(j, i)
                else:
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.TAIL.value:
                if mark_ji == Mark.ARROW.value:  # i <-- j
                    self.add_directed_Edge(j, i)
                elif mark_ji == Mark.CIRCLE.value: # i o-- j
                    self.add_circ_tail_Edge(i, j)
                elif mark_ji == Mark.TAIL.value:  # i --- j
                    self.add_tail_Edge(i, j)
                else:
                    raise ValueError(f"Invalid edge mark: {mark_ji} between {i} and {j}.")
            elif mark_ij == Mark.NULL.value:
                pass  # No edge between i and j
            else:
                raise ValueError(f"Invalid edge mark: {mark_ij} between {i} and {j}." "Invalid edge mark in the adjacency matrix.")

    def DAG_from_numpy_array(self, adj_matrix: np.ndarray, node_list: List[Node] = None):
        """
        Create a Directed Acyclic Graph (DAG) from a NumPy adjacency matrix.
        """
        if node_list is None:
            self.node_list = [Node(name=str(i), index=i) for i in range(adj_matrix.shape[0])]  # [Node(str, 0), Node(str, 1), ...]
        else:
            self.node_list = node_list


        self.add_nodes_from(self.node_list)
        for i, j in combinations(self.node_list, 2):
            mark_ij = adj_matrix[i.index, j.index]  # i markji --- markij j
            mark_ji = adj_matrix[j.index, i.index]


            if mark_ij == 1:
                self.add_directed_Edge(i, j)     # i --> j
            elif mark_ij == 0:
                pass  # No edge between i and j
            else:
                raise ValueError(f"Invalid edge mark: {mark_ij} between {i} and {j}." "Invalid edge mark in the adjacency matrix.")

    def from_pandas_adjacency(self, adj_matrix: pd.DataFrame, graph_type: str = 'MG'):

        """
        Create a Mixed Graph from a Pandas DataFrame adjacency matrix.
        """

        Node_list = [Node(name=node, index=i) for i, node in enumerate(adj_matrix.columns.to_list())]  # [Node
        if graph_type == 'MG':
            self.from_numpy_array(adj_matrix.to_numpy(), node_list=Node_list)
        elif graph_type == 'DAG':
            self.DAG_from_numpy_array(adj_matrix.to_numpy(), node_list=Node_list)


    def from_node_list(self, node_list: List[Union[str, Node]]):
        """
        Create a empty graph from a list of nodes.
        """
        if all(isinstance(node, Node) for node in node_list):
            # If node_list is already a list of Node objects, use it directly
            Node_list = node_list
        else:
            Node_list = [Node(name=node, index=i) for i, node in enumerate(node_list)] # [Node(str, 0), Node(str, 1), ...]
        self.node_list = Node_list
        self.add_nodes_from(self.node_list)
        self.clear_cache()

    ###############################################
    ## End Initializing Graphs in Different Ways ##
    ###############################################


    ####################
    ## Node functions ##
    ####################
    def add_Node(self, node: Node):
        """
        Add a node to the graph.

        Parameters:
        - node: The node to be added.
        """
        if not isinstance(node, Node):
            raise TypeError("node must be of type Node.")
        self.add_node(node)
        self.node_list.append(node)
        # Keep node_list ordered by node.index (ascending)
        self.node_list.sort(key=lambda n: n.index if n.index is not None else 0)


    def get_adj_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent to the given node in the graph.
        *-* node
        """
        t = node.index
        nodes_by_src = self._cache_nodes['adj_nodes']   # If not exists, automatically create an empty dict
        adj_nodes = nodes_by_src.setdefault(t, set())
        if adj_nodes:  # Cache exists and the cached set is not empty
            return adj_nodes.copy()

        adj = set(self.neighbors(node))
        nodes_by_src[t] = adj

        return adj.copy()

    def get_into_nodes(self, node: Node) -> Set[Node]:
        """
        Get the set of nodes that are adjacent into the given node in the graph.
        node <-*
        """
        t = node.index
        nodes_by_src = self._cache_nodes['into_nodes']   # If not exists, automatically create an empty dict
        into_nodes_t = nodes_by_src.setdefault(t, set())
        if into_nodes_t:  # Cache exists and the cached set is not empty
            return into_nodes_t.copy()

        adj_nodes = self.get_adj_nodes(node)
        into_nodes = set()
        for adj_node in adj_nodes:
            if self.has_into_Edge(adj_node, node):
                into_nodes.add(adj_node)

        nodes_by_src[t] = into_nodes
        return into_nodes.copy()


    def find_unique_triplets(self) -> List[Tuple[Node, Node, Node]]:
        """
        Efficiently find unique triplets <z, y, x> in MixGraph, avoiding symmetric duplicates.
        z.index < y.index < x.index, avoid <z, y, x> and <x, y, z>
        Returns:
            List [Tuple[Node, Node, Node]]: List of unique triplets.
        """


        triplets = []

        for y in self.node_list:
            # Get neighbors of y
            neighbors_y = self.get_adj_nodes(y)

            if len(neighbors_y) >= 2:
                for z, x in combinations(neighbors_y, 2):
                    triplets.append((z, y, x))


        return triplets


    ########################
    ## End Node functions ##
    ########################

    def _init_complete_graph(self):
        """
        Initialize a complete undirected graph with the given node set.
        """
        for node1, node2 in combinations(self.node_list, 2):
            self.add_circ_Edge(node1, node2)
        self.clear_cache()

    #############################
    ## Visualization functions ##
    #############################
    """
    Install Graphviz from https://graphviz.org/download/ (12.2.1)
    and make sure to add the Graphviz bin directory to your PATH environment variable.
    Use 'dot -V' in the command line to check if Graphviz is installed correctly.
    NOTE: (1) pip install graphviz==0.20.3 pydot==3.0.2
          (2) add the graphviz bin directory (default path: C:\Program Files\Graphviz\bin) to your PATH environment variable

    The function to_pydot() is better than to_dot() in using.
    """


    #################################
    ## End Visualization functions ##
    #################################

    #####################
    ## Graph functions ##
    #####################


    def copy(self) -> "MixGraph":
        """
        Create a deep copy of the MixGraph, including the node_list attribute.
        """
        new_graph = MixGraph()
        # Copy the graph structure from networkx
        new_graph.add_nodes_from(self.nodes())
        new_graph.add_edges_from((u, v, attr.copy()) for (u, v, attr) in self.edges(data=True))
        # Copy the node_list
        new_graph.node_list = self.node_list.copy() if hasattr(self, 'node_list') else []
        # Copy the cache
        new_graph._cache_path = self._cache_path.copy()
        new_graph._cache_nodes = self._cache_nodes.copy()
        return new_graph

    #########################
    ## End Graph functions ##
    #########################

