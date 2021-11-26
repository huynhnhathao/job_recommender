# A Graph-based Hybrid Job Recommender

*Attention: Do not mind the `Julia` language on the right, I don't use the Julia language in this project, it's the jsonline file extension `.jl`(jsonline is just like json but each line is a json object.) that makes github think I used Julia. This project is written in Python.*

My implementation of [A Recommender System for Job Seeking and Recruiting Website](http://www2013.w3c.br/companion/p963.pdf) using Scrapy, Networkx, Streamlit.

### Problem
The problem we are trying to solve in this project is: given a candidate's CV and/or interactions data of that candidate to the recommender system, which jobs and employers should the candidate suggest to the user?

## Four main steps

1. **Graph construction**. Model the inter-relations between entities in a heterogeneous and multi-relational directed graph. The nodes are employers, jobs and candidates, and the edges are inter-relations between them. Bi-directional relations are translated into two directed edges, for example, `similar` between two candidates can be represented by 2 directed `similar` edges between them. If the relations are bi-directional but heterogeneous, for example employer and her posted job, we will represent it with 2 different directed edges, from employer to job is `posted` edge and the other edge is `posted by` edge.
2.  **Context definition**. If the candidate search for jobs/employers using a keyword, then every nodes matched that keyword is the context at hand. If the candidate choosing an employer, then every node relates to the employer node become the context. The algorithm is expected to bias results towards the context in such a way that entities that have strong connections to contextual nodes get important ranking, this will lead to suggesting new relevant entities to the target actor depending on the strength of their connection to contextual nodes.
3. **Importance calculation**. Once the graph is formed and the context is defined, the 3A ranking algorithm is applied. *The ideas is:  a node is recursively important to a particular root set of nodes(representing the target user and the context) if and only if many important nodes connected to the root set via important edge types point to it.*
4. **Rank list extraction**. A list of entities ordered by their ranking importance will be recommended to the user.

## PageRank algorithm

Imagine a random walker who randomly visit websites to websites, the next website this random walker is going to visit depend on whether it is linked in the current website. In this model, websites are nodes and links are edges, if a website point to another website by citing it, there will has an edge between the citing website to the cited website. Now imagine that the random walker will walk a billion times, then we count the number of times he visit each node in the network. The rank of a node is the relative number of times that random walker visit that node. We can obtain the probability of a random walker visit a node by divide the number of times he visit that node to all of his visiting times. We need two modification for this model to work:

- If a node does not link to any one, then it is a dead end, we have to point that node to all the node in the network including itself, then the probability of moving from that node to other node is $\frac{1}{n}$
- If a group of node is isolated and does not has any link out of that group, then it is a dead end component. If the random walker fall into this component, he will never can see light again, as a result, all the probability will concentrate into that group. We need to define a teleportation probability to any other node in the network. At each transition, the random surfer may either jump to an arbitrary page with probability α, or follow one of the links on the page with probability (1 − α). $\alpha$ is also known as damping probability.



## Personalized PageRank

The key idea is using the teleportation probability to bias toward the context nodes. 

User can choose which skills he has, then the search results respond to him will bias toward these skill, expertise

Given a target node $i_q$ and a subset of nodes $S ⊆ N$ from graph $G = (N, A)$, rank the
nodes in $S$ in their order of similarity to $i_q$.

  That means the closest nodes to the target node will have the highest ranking, but we don't want to show the user nodes that are already familiar with the user, so we can filter out those node to introduce new nodes to user. Concretely, we will only filter out jobs that was connect to user by `apply` edge. 

### Ideas
The idea is we build a directed weighted network of nodes and edges, where nodes are entities and edges are relations between entities. There are three types of entities: candidates, employers and jobs. We used content-based and interaction-based between nodes to create relations between entities. When there is a new user login into our system, we will ask her to provide her information, such as CV, favorite/like employers and jobs, applied jobs. The first requirement is natural to ask a new user, but the later is not, if you provide the later information, then the system will pretend that you are an old user and it has your history interaction data. But if you don't provide that information, the system will treat you as a new user.

