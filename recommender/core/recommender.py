from typing import List

import numpy as np
import pandas as pd
import networkx as nx


from latent_semantic_analysis import *

class JobRecommender:
    def __init__(self, G: nx.MultiDiGraph, 
                lsa: LSA, 
                ):
        self.G = G
        self.lsa = lsa

    def _search(self, keywords: str) -> list:
        """Search for all jobs that match keywords"""
        pass

    def _personalized_rank_nodes(self, target_node: str,
                        context_nodes: List[str]) -> List[str]:
        """Rank a list of nodes using personalized PageRank w.r.t target node 
        and context nodes
        """
        pass

