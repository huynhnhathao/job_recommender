# A Graph-based Hybrid Job Recommender

*Attention: Do not mind the `Julia` language on the right, I don't use the Julia language in this project, it's the jsonline file extension `.jl`(jsonline is just like json but each line is a json object.) that makes github think I used Julia. This project is written in Python.*

My implementation of [A Recommender System for Job Seeking and Recruiting Website](http://www2013.w3c.br/companion/p963.pdf) using scrapy, networkx, streamlit.

The idea is we build a directed weighted network of nodes and edges, where nodes are entities and edges are relations between entities. There are three types of entities: candidates, employers and jobs. We used content-based and interaction-based between nodes to create relations between entities. When there is a new user login into our system, we will ask her to provide her information, such as CV, favorite/like employers and jobs, applied jobs. The first requirement is natural to ask a new user, but the later is not, because I don't want the compliated works of creating such an interaction mechanism for users to interact with the system and collect those interation data, so I will assume that these data are provided.
