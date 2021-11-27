from typing import Dict, List, Tuple, Any
import logging
import os
import pickle
from networkx.algorithms.shortest_paths import weighted

import numpy as np
import pandas as pd
import networkx as nx

from sklearn import neighbors

import latent_semantic_analysis
import constants

handler = logging.StreamHandler()
formmater = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class NetworkBuilder:
    """This class build a network of jobs, employers and candidates.

    First, all the entities of three types will be added to one single network
    as nodes. Each node has attribute `type`: str to tell which type it is.

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
        elif isinstance(employers_data, str):
            self.employers_data = pd.read_csv(employers_data)
            self.employers_data = self.employers_dataframe_to_dict(self.employers_data)
        else:
            raise ValueError('data should be a dataframe of a path to that dataframe')

        if isinstance(jobs_data, pd.DataFrame):
            self.jobs_data = self.jobs_dataframe_to_dict(jobs_data)
        elif isinstance(jobs_data, str):

            self.jobs_data = pd.read_csv(jobs_data)
            self.jobs_data = self.jobs_dataframe_to_dict(self.jobs_data)
        else:
            raise ValueError('data should be a dataframe of a path to that dataframe')

        if isinstance(cv_data, pd.DataFrame):
            self.cv_data = self.cv_dataframe_to_dict(cv_data)
        elif isinstance(cv_data, str):
            self.cv_data = pd.read_csv(cv_data)
            self.cv_data = self.cv_dataframe_to_dict(self.cv_data)
        else:
            raise ValueError('data should be a dataframe of a path to that dataframe')

        # Create the master jobs network
        self.G = None

        # Create a text comparer using latent_semantic_analysis
        self.comparer = None

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
        
        # graph counts and saves its nodes and edges types
        G = nx.MultiDiGraph(name = 'Jobs graph',
                            num_employers = 0, 
                            num_jobs = 0,
                            num_candidates = 0,
                            candidate_to_job = 0,
                            candidate_to_candidate = 0,
                            job_to_job = 0,
                            employer_to_employer = 0,
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
        
        
        return G

    def get_all_document_from_graph(self) -> List[str]:
        """This method will extract all documents attribute of every node in G,
        used to construct the vocab for LSA
        """
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

    def get_lsa(self,) -> latent_semantic_analysis.LSA:
        """Create a LSA object, which will be used to compare documents.
        """

        
        # if lsa object exist, load it
        if os.path.isfile(constants.LSA_COMPARER_PATH):
            logger.info(f'Loading LSA comparer from {constants.LSA_COMPARER_PATH}')
            with open(constants.LSA_COMPARER_PATH, 'rb') as f:
                self.lsa = pickle.load(f,)
        else:


            all_documents = self.get_all_document_from_graph()
            all_texts = ' '.join(all_documents)

            vocab = latent_semantic_analysis.make_vocab(all_texts, min_word_count=10)

            self.lsa = latent_semantic_analysis.LSA(vocab, all_documents, constants.NUM_REDUCED_FEATURES)
            self.lsa.do_work()
            logger.info(f'Saving LSA comparer to {constants.LSA_COMPARER_PATH}')
            with open(constants.LSA_COMPARER_PATH, 'wb') as f:
                pickle.dump(self.lsa, f, pickle.HIGHEST_PROTOCOL )


    def vectorize_nodes(self) -> None:
        """
        For each node in self.G, add a node attribute `tfidf` represent the node 
        content as a reduced tf-idf vector.
        """
        # loop over all node
        for node_name in self.G:
            node = self.G.nodes[node_name]
            type = node['node_type']
            document = None
            if type == 'employer':
                document = [str(node['overview']), str(node['expertise']), str(node['benifit'])]
            elif type == 'job':
                document = [str(node['three_reasons']), str(node['description'])]
            elif type == 'candidate':
                document = [str(node['expertise']), str(node['resume'])]
            if document is None:
                logger.info(f'Node {node} has no information to vectorize')
                self.G.nodes[node_name]['reduced_tfidf'] = np.array([])
                continue
            try:
                document = ' '.join(document)
            except:
                print(document)
            vector = self.lsa.vectorize(document)
            self.G.nodes[node_name]['reduced_tfidf'] = vector

    def get_k_neighbors(self, data: np.ndarray, label: np.ndarray,
            k: int) -> Tuple[neighbors.KNeighborsClassifier, List[List[int]]]:
        """Build a KNN classifier, fit on data and find K neighbors for each
        data point in data
        Args
            data: ndarray of shape (n_samples, dim) data to fit KNN
            num_neighbors: number of neighbors to find for each point
        Returns
            The fitted KNN object, a list of list of neighbors of each data point
            the returned neighbors are just the indices of data points in our data.
        """
        knn = neighbors.KNeighborsClassifier(10, metric = 'euclidean')
        knn.fit(data, label)
        all_neighbors = []
        for point in data:
            nbs = knn.kneighbors(point.reshape(1, -1), k, return_distance=False)
            all_neighbors.append(nbs)


        return knn, all_neighbors

    def add_relations_edges(self, method = 'knn') -> None:
        """This method use Latent semantic analysis to compare different types
        of nodes to infer the 'similar' relation between them and add those edges
        to the network
        First, for each type of entities, computes and infers the 'similar' relations
        between them and add edges represent those relations.
        Second, for each candidate, compute the 'profile match' relation between 
        them and add edges represent those relation

        Args:
            method: can be `knn` or `cosine`, use knn classifier or cosine similarity
                to find neighbors
        """
        
        # Each type of entity has its own knn model
        employer_node_names = [key for key, item in self.G.nodes.items() if item['node_type'] == 'employer']
        job_node_names = [key for key, item in self.G.nodes.items() if item['node_type'] == 'job']
        candidate_node_names = [key for key, item in self.G.nodes.items() if item['node_type'] == 'candidate']

        employer_vectors = [self.G.nodes[employer]['reduced_tfidf'] for employer in employer_node_names]
        employer_vectors = np.array(employer_vectors).reshape(len(employer_node_names), -1)
        num_employer_neighbors = int(constants.NEIGHBOR_RATIO * len(employer_node_names))
        self.employer_knn, employer_neighbors = self.get_k_neighbors(employer_vectors,
                                                np.arange(len(employer_node_names)),
                                                num_employer_neighbors)


        job_vectors = [self.G.nodes[job]['reduced_tfidf'] for job in job_node_names]
        job_vectors = np.array(job_vectors).reshape(len(job_node_names), -1)
        num_job_neighbors = int(constants.NEIGHBOR_RATIO * len(job_node_names))
        self.job_knn, job_neighbors = self.get_k_neighbors(job_vectors,
                                    np.arange(len(job_node_names)),
                                    num_job_neighbors)


        candidate_vectors = [self.G.nodes[candidate]['reduced_tfidf'] for candidate in candidate_node_names]
        candidate_vectors = np.array(candidate_vectors).reshape(len(candidate_node_names), -1)

        num_candidate_neighbors = int(constants.NEIGHBOR_RATIO * len(candidate_node_names))

        self.candidate_knn, candidate_neighbors = self.get_k_neighbors(candidate_vectors,
                                                np.arange(len(candidate_node_names)),
                                                num_candidate_neighbors)


        # counting the number of edges for each edge type
        for i, nbs in enumerate(employer_neighbors):
            this_employer = employer_node_names[i]
            for nb_index in nbs[0]:
                try:
                    nb_name = employer_node_names[nb_index]
                except:
                    print(nb_index)
                if not self.G.has_edge(this_employer, nb_name):
                    self.G.add_edge(this_employer, nb_name,
                            edge_type = 'employer_to_employer',
                            weight = constants.SIMILAR_WEIGHT)
                    self.G.graph['employer_to_employer'] += 1
                

        for i, nbs in enumerate(job_neighbors):
            this_job = job_node_names[i]
            for nb_index in nbs[0]:
                nb_name = job_node_names[nb_index]
                if not self.G.has_edge(this_job, nb_name):
                    self.G.add_edge(this_job, nb_name,
                            edge_type = 'job_to_job',
                            weight = constants.SIMILAR_WEIGHT)
                    self.G.graph['job_to_job'] += 1

        for i, nbs in enumerate(candidate_neighbors):
            this_candidate = candidate_node_names[i]
            for nb_index in nbs[0]:
                nb_name = candidate_node_names[nb_index]
                if not self.G.has_edge(this_candidate, nb_name):
                    self.G.add_edge(this_candidate, nb_name,
                            edge_type = 'candidate_to_candidate',
                            weight = constants.SIMILAR_WEIGHT)
                    self.G.graph['candidate_to_candidate'] += 1
        # now add candidate_to_job or profile_matched edges

        pm_num_neigbors = int(constants.PROFILE_MATCHED_NEIHBOR_RATIO * len(candidate_node_names))
        for i, vector in enumerate(candidate_vectors):
            this_candidate = candidate_node_names[i]
            nbs = self.job_knn.kneighbors(vector.reshape(1, -1), pm_num_neigbors, return_distance = False)
            for nb_index in nbs[0]:
                nb_name = job_node_names[nb_index]
                if not self.G.has_edge(this_candidate, nb_name):
                    self.G.add_edge(this_candidate, nb_name,
                            edge_type = 'candidate_to_job',
                            weight = constants.PROFILE_MATCH_WEIGHT)
                    self.G.graph['candidate_to_job'] +=1


        # if two candidate have the same expertise, they are connected to each
        # others
        for c1 in candidate_node_names:
            c1_expertise = self.G.nodes[c1]['expertise'] 
            for c2 in candidate_node_names:
                if c1 == c2:
                    continue
                else:
                    if self.G.nodes[c2]['expertise'] == c1_expertise:
                        if not self.G.has_edge(c1, c2):
                            self.G.add_edge(c1, c2,
                                    edge_type = 'candidate_to_candidate',
                                    weight = constants.SIMILAR_WEIGHT)
                            self.G.graph['candidate_to_candidate'] += 1


    def build(self,) -> None:
        """Build the network.
        """

        logger.info('Start building the master Graph...')
        self.G = self.create_network_from_data()
        logger.info('Master Graph is built.')

        logger.info('Creating LSA comparer')
        self.get_lsa() 
        
        logger.info('Compute tf-idf vector for every node')
        self.vectorize_nodes()
        
        logger.info('Adding relations edges...')
        self.add_relations_edges()

        logger.info('Network is built.')


if __name__ == '__main__':

    if os.path.isfile(constants.NETWORK_BUILDER_SAVE_PATH):
        logger.info(f'Loading network from {constants.NETWORK_BUILDER_SAVE_PATH}')
        with open(constants.NETWORK_BUILDER_SAVE_PATH, 'rb') as f:
            network = pickle.load(f)

    else:
        network = NetworkBuilder(constants.EMPLOYERS_DATA_PATH,
                            constants.JOBS_DATA_PATH,
                            constants.CV_DATAPATH)

        network.build()

        logger.info(f'saving network builder object to {constants.NETWORK_BUILDER_SAVE_PATH}')

        with open(constants.NETWORK_BUILDER_SAVE_PATH, 'wb') as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
