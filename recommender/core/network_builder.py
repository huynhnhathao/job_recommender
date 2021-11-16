from typing import Dict, List, Any
import logging

import numpy as np
import pandas as pd
import networkx as nx

from recommender.core import latent_semantic_analysis

handler = logging.StreamHandler()
formmater = logging.Formatter(r'%(asctime)s - %(message)s')
handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    def __init__(self, employers_data: pd.DataFrame, 
                jobs_data: pd.DataFrame,
                cv_data: pd.DataFrame) -> None:
        """
        Args:
            employer_data: a dict map from employer/company unique id to its data
            job_data: a dict map from job's unique id to its data
            cv_data: CV of candidates in the network
        All of those data are expected to be in English

        """
        if isinstance(employers_data, pd.DataFrame):
            self.employers_data = self.employers_dataframe_to_dict(employers_data)
        else:
            self.employers_data = employers_data

        if isinstance(jobs_data, pd.DataFrame):
            self.jobs_data = self.jobs_dataframe_to_dict(jobs_data)
        else:
            self.jobs_data = jobs_data

        if isinstance(cv_data, pd.DataFrame):
            self.cv_data = self.cv_dataframe_to_dict(cv_data)
        else:
            self.cv_data = cv_data

        # Create the master jobs network
        self.G = self.create_network_from_data()

        # Create a text comparer using latent_semantic_analysis
        self.comparer = self.get_lsa() 

    def employers_dataframe_to_dict(self,
                        companies_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
                        
        # this method only transform dataframe to dict, it does not add any 
        # information into the transformed dict
        employers_data = {}
        for i, row in companies_df.iterrows():
            employer_data = {'company_name': row['company_name'],
                        'average_rating': row['average_rating'],
                        'num_review': row['num_review'],
                        'city': row['city'],
                        'type': row['type'],
                        'num_employee': row['num_employee'],
                        'country': row['country'],
                        'working_day': row['working_day'],
                        'OT': row['OT'],
                        'overview': row['overview'],
                        'expertise': row['expertise'],
                        'benifit': row['benifit'],
                        'logo_link': row['logo_link']}
            
            employers_data[row['company_id']] = employer_data

        return employers_data

    def jobs_dataframe_to_dict(self,
                            jobs_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:

        # this method only transform dataframe to dict, it does not add any 
        # information into the transformed dict
        jobs_data = {}
        for i, row in jobs_df.iterrows():
            job_data = {'company_id': row['company_id'],
                    'job_name': row['job_name'],
                    'taglist': row['taglist'],
                    'location': row['location'], 
                    'three_reasons': row['three_reasons'],
                    'description': row['description']}
            jobs_data[row['job_id']] = job_data
                                        
        return jobs_data

    def cv_dataframe_to_dict(self, 
                            cv_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:

        # this method only transform dataframe to dict, it does not add any 
        # information into the transformed dict
        cv_data = {}
        for i, row in cv_df.iterrows():
            
            cv_data[i] = {'expertise': row['Category'],
                        'resume': row['Resume'] }

        return cv_data

    def create_network_from_data(self, ) -> nx.MultiDiGraph:
        """
        Create a networkx MultiDiGraph to represent employers, jobs, candidates
        as nodes and the 'posted' relations between employer and job as edge.

        The graph created by this method only create edges that can be directly 
        infered from data, such as 'posted' and interaction edges.
        Other types of edges that need to compare objects will be added in another
        method
        """
        logger.info('Start building the master Graph...')
        # graph counts and saves its nodes and edges types
        G = nx.MultiDiGraph(name = 'Jobs graph',
                            num_employers = 0, 
                            num_jobs = 0,
                            num_candidates = 0,
                            num_candidate_match_job = 0,
                            num_similar_candidates = 0,
                            num_similar_jobs = 0,
                            num_similar_employers = 0,
                            num_apply = 0,
                            num_favorite = 0)

        # first add all employer nodes to the network and its data
        for employer_id, employer_data in self.employers_data.items():
            G.add_node(employer_id, node_type = 'employer', **employer_data)
            G.graph['num_employers'] += 1
            
        # add all job nodes and the bidirectional edges from job node to employer node
        for job_id, job_data in self.jobs_data.items():
            G.add_node(job_id, node_type = 'job', **job_data)
            G.graph['num_jobs'] += 1
            # add two edges between job and its employers\
            G.add_edge(job_id, job_data['company_id'], weight = 1, 
                        edge_type = 'posted')
            G.add_edge(job_data['company_id'], job_id, weight = 1, 
                        edge_type = 'posted')

        # add candidates to the network. Candidate node ids are set and managed
        # by the network

        for _, candidate_data in self.cv_data.items():
            candidate_id = 'candidate-%d'%G.graph['num_candidates']
            G.add_node(candidate_id, node_type = 'candidate', **candidate_data)
            G.graph['num_candidates'] += 1
        
        logger.info('Master Graph is built.')
        return G

    def add_relations_edges(self) -> None:
        """This method use Latent semantic analysis to compare different types
        of nodes to infer the 'similar' relation between them and add those edges
        to the network
        """

    def get_all_document_from_graph(self) -> List[str]:
        """This method will extract all documents attribute of every node in G"""
        all_documents = []
        for _, node_data in self.G.nodes.items():
            if not node_data:
                print(_)
                break
            if node_data['node_type'] == 'employer':
                all_documents.append(' '.join([str(node_data['overview']), str(node_data['benifit']) ]))
            elif node_data['node_type'] == 'job':
                all_documents.append(' '.join([str(node_data['three_reasons']), str(node_data['description']) ]))
            elif node_data['node_type'] == 'candidate':
                all_documents.append(node_data['resume'])
            else:
                continue
        
        return all_documents

    def get_lsa(self,) -> Any:
        """Create a LSA object, which will be used to compare documents.
        """
        logger.info('Creating Comparer')
        all_documents = self.get_all_document_from_graph()
        all_texts = ' '.join(all_documents)

        logger.info('Creating vocab')
        vocab = latent_semantic_analysis.make_vocab(all_texts, min_word_count=10)
        logger.info('Vocab is created')

        lsa = latent_semantic_analysis.LSA(vocab)
        # elaborate the vocab choosing process such that no number and no nonsense
        #  word are chosen as vocabs 
