from typing import Dict, List, Any

import numpy as np
import pandas as pd
import networkx as nx

class NetworkBuilder:
    """This class build a network of jobs, employers and candidates.

    First, all the entities of three types will be added to one single network
    as nodes. Each node has attribute type: str to tell which type is it.

    Then all the relations that can be extracted directly from data will be added
    to the networks as edges. Those relations include a company posted a job.

    Then the similar relation between two node of the same type can be computed 
    using latent semantic analysis, the most k percent similar nodes will be connected
    by the 'similar' edge.

    The next relation is 'profile match', which is a relation between a candidate 
    and a job. We can derrive this relation using LSA as above.

    Another interaction relations can be added from data are apply, favorite, 
    like and visit. We will consider simulate it.

    The network only contains the semantic relationships beween nodes, not the 
    data of these node. We will use the entity's id to represent an entity.



    Node types:
        Employer: str: a unique string id for an employer
        Candidate: str: a unique string id for a candidate
        Job: str: a unique string id for a job posted by an employer

    
    """
    def __init__(self, employers_data: Dict[str, Dict[str, str]], 
                jobs_data: Dict[str, Dict[str, str]]) -> None:
        """
        Args:
            employer_data: a dict map from employer/company unique id to its data
            job_data: a dict map from job's unique id to its data

        """
        self.employers_data = employers_data
        self.jobs_data = jobs_data

        self.network = self.create_network_from_data()

    def create_network_from_data(self, ) -> nx.MultiDiGraph:
        # each edge and node will associate with its data, which is a dict.
        # but the nodes and edges itself are just unique identifier
        network = nx.MultiDiGraph()
        
        