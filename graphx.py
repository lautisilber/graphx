from platform import node
from tkinter.messagebox import NO
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Union

Nodes_T = Union(list, tuple, np.ndarray)
Edges_T = Union(list, tuple, np.ndarray)





class GraphBase(ABC):
    '''
        For now graphs are immutable once they are created
    '''
    def __init__(self, edges: Edges_T) -> None:
        '''
            edges is a list of 1-dim len=2 lists
        '''
        for edge in edges:
            if not len(edge) == 2:
                raise Exception("variable 'edges' has to be a 1-dim list of len=2 lists")
        
        self.nodes = set([node for edge in edges for node in edge])
        self.nodes_n = {node:i for i, node in enumerate(self.nodes)}
        self.adj = np.matrix(0)

        # subclasses need to implement adj matrix creation

    def get_degrees(self) -> np.ndarray:
        return self.adj @ np.ones(self.adj.shape[1])

    def get_node_degree(self, node) -> int:
        return self.get_degrees[self.nodes_n[node]]


class Graph(GraphBase):
    '''
        undirected unweighted graph
    '''
    def __init__(self, edges: Edges_T) -> None:
        super().__init__(edges)
        self.adj = np.matrix(np.zeros([len(self.nodes)]*2), dtype=np.uint8)

        for edge in edges:
            i = self.nodes_n[edge[0]]
            j = self.nodes_n[edge[1]]
            self.adj[i,j] = 1



