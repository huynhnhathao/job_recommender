import scrapy
import json


class JobSpider(scrapy.Spider):
    """This spider will crawl all relevant information of all jobs for all 
    company in companies_info.jl
    
    A job will have the following information:
        company_name: str: the company name that offer this job
        job_name: str
        programing_languages: str: required programing languages
        salary: str: either a range or not revealed
        location: str: location of the job
        time_posted: str: how long it was posted
        three_reasons: str: Top three reasons to join us
        description: str
        requirements: str: your skills and Experience
        why_join: str: why you'll love working here

    
    """

    # start_urls = []
    # file_path = '/home/azureuser/cloudfiles/code/Users/job_recommender/crawl_data/companies_info.jl'
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         jobs = json.loads(line)['jobs']
    #         if len(jobs):
    #             # start_urls.append(url)
