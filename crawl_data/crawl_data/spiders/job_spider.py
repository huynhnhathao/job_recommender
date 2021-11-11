import scrapy
import json


class JobSpider(scrapy.Spider):
    """This spider will crawl all relevant information of all jobs for all 
    company in companies_info.jl
    
    A job will have the following information:
        company_name: str: the company name that offer this job
        job_name: str
        tag_list: str: tag list of the job
        location: str: location of the job
        three_reasons: str: Top three reasons to join us
        description: str
        requirements: str: your skills and Experience
        why_join: str: why you'll love working here

    
    """
    name = 'job_info'

    def start_requests(self, ):
        
        company_jobs = {}
        file_path = r'C:\Users\ASUS\Desktop\job_recommender\crawl_data\companies_info.jl'
        with open(file_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                company_info = json.loads(line)
                if company_info['company_name']:
                    company_jobs[company_info['company_name']] = company_info['jobs']

        for company_name, jobs in company_jobs.items():
            for job_name, job_url in jobs.items():
                yield scrapy.Request(job_url,
                    cb_kwargs = {'company_name': company_name, 'job_name': job_name},
                    callback= self.parse)

    def parse(self, response, company_name, job_name):
        job_info = {}

        job_info['company_name'] = company_name
        job_info['job_name'] = job_name
        job_info['tag_list'] = '\n'.join(response.xpath("//div[@class='job-details__tag-list']/a/span/text()").getall())
        job_info['location'] = response.xpath("//div[@class='job-details__overview']//div[@class='svg-icon__text']//span/text()").get()
        job_info['three_reasons'] = '\n'.join(response.xpath("//div[@class='job-details__top-reason-to-join-us']//li/text()").getall())

        job_info['description'] = '\n'.join(response.xpath("//div[@class='job-details__paragraph']").getall())
        
        yield job_info

        