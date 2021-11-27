# EMPLOYERS_DATA_PATH = r'C:\Users\ASUS\Desktop\repositories\job_recommender\data\companies.csv'
# JOBS_DATA_PATH = r'C:\Users\ASUS\Desktop\repositories\job_recommender\data\jobs.csv'
# CV_DATAPATH = r'C:\Users\ASUS\Desktop\repositories\job_recommender\data\cvdata\ResumeDataSet.csv'

LOG_FILE_PATH = '/home/huynhhao/Desktop/job_recommender/recommender/core/log.txt'
EMPLOYERS_DATA_PATH = '/home/huynhhao/Desktop/job_recommender/data/companies.csv'
JOBS_DATA_PATH = '/home/huynhhao/Desktop/job_recommender/data/jobs.csv'
CV_DATAPATH = '/home/huynhhao/Desktop/job_recommender/data/cvdata/ResumeDataSet.csv'
NETWORK_DATA_PATH = '/home/huynhhao/Desktop/job_recommender/data/network_data'
NETWORK_BUILDER_SAVE_PATH = '/home/huynhhao/Desktop/job_recommender/data/network_data/network_builder.pkl'

VOCAB_PATH = '/home/huynhhao/Desktop/job_recommender/data/network_data/vocab.json'
LSA_COMPARER_PATH = '/home/huynhhao/Desktop/job_recommender/data/network_data/lsa.pkl'


POSTED_WEIGHT = 1
APPLIED_WEIGHT = 2
SIMILAR_WEIGHT = 1.5
PROFILE_MATCH_WEIGHT = 1.5
FAVORITE_WEIGHT = 1
LIKE_WEIGHT  = 0.5
VISIT_WEIGHT = 0.2


NUM_REDUCED_FEATURES = 0.3
# How many neighbors should one node connect to, if it's float, it's the ratio
# of all node considered. 
NEIGHBOR_RATIO = 0.01
PROFILE_MATCHED_NEIHBOR_RATIO = 0.01