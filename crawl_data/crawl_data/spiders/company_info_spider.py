import scrapy
import json

class CompanyInfoSpider(scrapy.Spider):
    """
    This spider will loop over all the company link saved previously by the
    Companies spider and crawl all the relevant info of that company and save to 
    a json line file

    A company will has following information:
        name: str: name of the company
        location: str: location of the company
        type: str: type of company, i.g product, ...
        num_employee: str: number of employee
        country: str:  country of the company
        working_day: str: from _ to _ working day
        OT: bool: has overtime or not
        overview: str: overview of the company
        all_job: str: links to all the job posted by the company
    """

    name = 'company_info'
    start_url = []
    with open('/home/azureuser/cloudfiles/code/Users/job_recommender/crawl_data/companies_url.jl', 'r') as f:
        for line in f:
            start_url.append(json.loads(line))


    def parse(self, response):
        company_info = {}
        company_info['name'] = response.xpath("//h1[@class='headers__info__name']/text()").get().strip()
        
        header_info = response.xpath("//div[@class='svg-icon__text']/text()").getall()
        
        company_info['location'] = header_info[0].strip() if header_info else None
        company_info['type'] = header_info[1].strip()  if 1 < len(header_info) else None
        company_info['num_employee'] = header_info[2].strip() if 2 < len(header_info) else None
        company_info['country'] = header_info[3]  if 3 < len(header_info) else None
        company_info['working_day'] = header_info[4] if 4 < len(header_info) else None
        company_info['OT'] = header_info[5] if 5 < len(header_info) else None

        company_info['overview'] = 
                        
        
                            
                    

   

