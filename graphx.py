import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Union

Nodes_T = Union(list, tuple, np.ndarray)
Edges_T = Union(list, tuple, np.ndarray)



class Graph:
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
        self.adj = np.matrix(np.zeros([len(self.nodes)]*2), dtype=np.uint8)

        for edge in edges:
            i = self.nodes_n[edge[0]]
            j = self.nodes_n[edge[1]]
            self.adj[i,j] = 1

    def node_name_to_number(self, name) -> int:
        return self.nodes_n[name]

    def nodes_number_to_name(self, number: int):
        return self.nodes[number]

    def get_degrees(self) -> np.ndarray:
        '''
            array of node degrees (index is node number)
        '''
        return self.adj @ np.matrix(np.ones(self.adj.shape[1])).T

    def get_node_degree(self, node: int) -> int:
        '''
            degree of node n
        '''
        return self.get_degrees[node]

    @property
    def M(self) -> int:
        '''
            number of links
        '''
        return int(np.sum(self.ady) / 2)
    
    @property
    def N(self) -> int:
        return self.ady.shape[0]
    
    def mean_degree(self) -> float:
        return self.M / self.N
    
    def density(self) -> float:
        # M_max / M
        return self.mean_degree() / (self.N - 1)

    def n_ways_from_a_to_b(self, n: int, a: int=None, b: int=None) -> Union(np.ndarray, int):
        a_n = np.linalg.matrix_power(self.ady, n)
        a_n = np.multiply(a_n, np.matrix(np.ones(self.ady.shape) - np.diag(self.N))) # igualo diagonal a 0 xq creo q no tiene info
        if a is None or b is None:
            return a_n
        return a_n[a,b]
        
    def max_degree(self) -> int:
        '''
            degree of node with max degree
        '''
        return np.max(self.get_degrees())

    def n_k(self, k: int=None) -> Union(dict, int):
        '''
            dictionary with structure k:# of nodes with degree k
        '''
        n_k = {}
        degrees = set(self.get_degrees())
        for deg in degrees:
            n_k[deg] = np.sum(self.get_degrees() == deg)
        if k is None:
            return n_k
        if k not in n_k:
            return 0
        return n_k[k]
    
    def degree_distribution(self, k: int=None) -> Union(dict, float):
        '''
            dictionary with structure k:probability of choosing a node with degree k
        '''
        p_k = {}
        degrees = set(self.get_degrees())
        for deg in degrees:
            p_k[deg] = np.sum(self.get_degrees() == deg) / self.N
        if k is None:
            return p_k
        if k not in p_k:
            return 0
        return p_k[k]

    def clustering(self, k: int=None) -> Union(np.ndarray, float):
        a_3 = np.linal.matrix_power(self.ady, 3)
        degs = self.get_degrees()
        c_i = np.diag(a_3) / (degs * (degs - 1))
    
    def mean_clustering(self) -> float:
        c_i = self.clustering()
        c = np.sum(c_i) / self.N
        return c
    
    def global_clustering(self):
        a_2 = np.linal.matrix_power(self.ady, 2)
        a_3 = np.linal.matrix_power(self.ady, 3)
        return np.trace(a_3) / np.sum(a_2 - np.diag(a_2))



        

