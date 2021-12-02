import sys
import pickle
import time
# sys.path.append(r'C:\Users\huynhhao\Desktop\job_recommender')
# sys.path.append(r'C:\Users\huynhhao\Desktop\job_recommender\recommender\core')

import streamlit as st
import pandas as pd
import numpy as np

from recommender.core import job_recommender
from recommender.core.network_builder import *
from recommender.core import latent_semantic_analysis

st.set_page_config(page_title='The Ultimate Jobs Recommender', page_icon=None,
    layout="centered", initial_sidebar_state="auto", menu_items=None)

st.header('The Ultimate Jobs Recommender')
st.sidebar.markdown("""### [How does this work?](https://github.com/huynhnhathao/job_recommender)""")
page = st.sidebar.radio('Choose Your Page', options=['Your Information', 'Search'], index = 0)

alpha = st.sidebar.slider('Damping Probability', min_value=0., max_value=1., value=0.5, step=0.1)
st.sidebar.markdown("""The larger the damping probability, the more personalized the results are to you. 
                    """)

@st.cache(allow_output_mutation  = True)
def load_recommender():
    graphpath =  'data/network_data/graph.pkl'
    lsapath = 'data/network_data/lsa.pkl'
    with open(graphpath, 'rb') as f:
        G = pickle.load(f)

    with open(lsapath, 'rb') as f:
        lsa = pickle.load(f)

    jrec = job_recommender.JobRecommender(G, lsa)

    return jrec

all_expertises = ['Java Developer', 'Testing', 'DevOps Engineer', 'Python Developer',
       'Web Designing', 'HR', 'Hadoop', 'Blockchain', 'ETL Developer',
       'Operations Manager', 'Data Science', 'Sales', 'Mechanical Engineer',
       'Arts', 'Database', 'Electrical Engineering', 'Health and fitness',
       'PMO', 'Business Analyst', 'DotNet Developer', 'Automation Testing',
       'Network Security Engineer', 'SAP Developer', 'Civil Engineer',
       'Advocate']

# start = time.time()
# end = time.time()
# st.text(f"model loaded in {(end - start)/60}")

jrec = load_recommender()
user_data = {}

if page == 'Your Information':
    user_expertises = st.selectbox('Choose your expertise', options=all_expertises, index = 3)
    if len(user_expertises) < 1:
        st.info('Please tell us your expertises')

    if len(user_expertises) > 0:
        
        user_resume = st.text_area(label = 'Enter your resume',
            placeholder="My research interest are Machine learning,...")
        if 1 < len(user_resume) < 100:
            st.info('Please give us a longer resume :(')
            
    user_data['expertise'] = user_expertises.strip()
    user_data['resume'] = user_resume
    if len(user_data['resume']) > 100:
        jrec.add_node_to_graph('candidate', user_data)

elif page == 'Search':
    # if no node has added to the network, means user
    # does not provide any information, recommend the most
    # popular jobs in the network
    if jrec.target_node is None:
        st.markdown("**Those are the most popular jobs in our network.**")
        st.json(jrec.rank_nodes(personalized = False,
                target_node = jrec.target_node,
                return_node_type='job',
                alpha = alpha))
    # st.text(jrec.target_node)
    else:    
        search_keywords = st.text_input('Search', placeholder='Keywords skill (Solidity, Python), Job title, ...')
        if len(search_keywords) > 3:
            # First, search for all node that match the keywords
            search_results = jrec.search(search_keywords)
            # Then rank the nodes personalized to the user node and the context node.
            personalized_results = jrec._rank_node_with_context(jrec.target_node, 
                                        search_results, alpha, 'job')
            c1, c2, c3, c4, c5 = True, False, False, False, False
            col1, col2, col3, col4, col5 = st.columns(5)
            st.markdown(f"Found {len(personalized_results)} jobs out of {jrec.G.graph['num_jobs']} jobs.")
            for key, value in personalized_results.items():
                job_node = jrec.G.nodes[key]
                company_id = job_node['company_id']
                logo = jrec.G.nodes[company_id]['logo_link']
                
                if c1:                
                    with col1:
                        st.image(logo, caption = job_node['job_name'], width=60, use_column_width = 'always' )
                        c1, c2, c3, c4, c5 = False, True, False, False, False
                        continue
                elif c2:
                    with col2:
                        st.image(logo, caption = job_node['job_name'], width=60, use_column_width = 'always' )
                        c1, c2, c3, c4, c5 = False, False, True, False, False
                        continue
                elif c3:
                    with col3:
                        st.image(logo, caption = job_node['job_name'], width=60, use_column_width = 'always' )
                        c1, c2, c3, c4, c5 = False, False, False, True, False
                        continue

                elif c4:
                    with col4:
                        st.image(logo, caption = job_node['job_name'], width=60, use_column_width = 'always' )
                        c1, c2, c3, c4, c5 = False, False, False, False, True
                        continue
                elif c5:
                    with col5:
                        st.image(logo, caption = job_node['job_name'], width=60, use_column_width = 'always' )
                        c1, c2, c3, c4, c5 = True, False, False, False, False
                        continue
            # st.json(personalized_results)
        else:
            st.info('Enter your search keywords')