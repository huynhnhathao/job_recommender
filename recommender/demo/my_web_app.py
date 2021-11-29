import sys
import pickle
sys.path.append(r'C:\Users\ASUS\Desktop\repositories\job_recommender')

import streamlit as st
import pandas as pd
import numpy as np

from recommender.core import job_recommender


st.set_page_config(page_title='The Utimate Jobs Recommender', page_icon=None,
    layout="centered", initial_sidebar_state="auto", menu_items=None)

@st.cache
def load_recommender():
    file_path = r'C:\Users\ASUS\Desktop\repositories\job_recommender\data\network_data\network_builder.pkl'
    with open(file_path, 'rb') as f:
        network_builder = pickle.load(f)

    jrec = job_recommender.JobRecommender(network_builder.G, network_builder.lsa)
    return jrec

    


keywords = st.text_input('Search for jobs, companies')
st.text(keywords)

if len(keywords) > 0:
    st.image('https://itviec.com/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBOUFuREE9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--4f2e6d9f3a3f8fd68828c39ac6b14b8ea961c55e/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdCem9MWm05eWJXRjBTU0lJY0c1bkJqb0dSVlE2RW5KbGMybDZaVjkwYjE5bWFYUmJCMmtCcWpBPSIsImV4cCI6bnVsbCwicHVyIjoidmFyaWF0aW9uIn19--623b1a923c4c6ecbacda77c459f93960558db010/dek-technologies-logo.png', caption='Hello', width=50, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
