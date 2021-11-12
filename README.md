# A Graph-based Hybrid Job Recommender

*Attention: Do not mind the `Julia` language on the right, I don't use the Julia language in this project, it's the jsonline file extension `.jl`(jsonline is just like json but each line is a json object.) that makes github think I used Julia. This project is written in Python.*

My implementation of [A Recommender System for Job Seeking and Recruiting Website](http://www2013.w3c.br/companion/p963.pdf) using scrapy, networkx, streamlit.

### Problem
The problem we are trying to solve in this project is: given a candidate's CV and/or interactions data of that candidate to the recommender system, which jobs and employers should the candidate suggest to the user?

### Ideas
The idea is we build a directed weighted network of nodes and edges, where nodes are entities and edges are relations between entities. There are three types of entities: candidates, employers and jobs. We used content-based and interaction-based between nodes to create relations between entities. When there is a new user login into our system, we will ask her to provide her information, such as CV, favorite/like employers and jobs, applied jobs. The first requirement is natural to ask a new user, but the later is not, if you provide the later information, then the system will pretend that you are an old user and it has your history interation data. But if you don't provide that information, the system will treat you as a new user.
