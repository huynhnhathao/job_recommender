import scrapy
import json

# TODO: This spider does not get the "Why you'll love working here", fix it.
class CompanyInfoSpider(scrapy.Spider):
    """
    This spider will loop over all the company link saved previously by the
    Companies spider and crawl all the relevant info of that company and save to 
    a json line file

    A company will has following information:
        company_name: code name of the company, can be used as id
        name: str: display name of the company
        logo: link to the logo image of the company
        location: str: location of the company
        type: str: type of company, i.g product, ...
        num_employee: str: number of employee
        country: str:  country of the company
        working_day: str: from _ to _ working day
        OT: bool: has overtime or not
        overview: str: overview of the company
        expertise: company's expertise
        benifit: employee's bennifit when working for employee
        jobs: dict[name: link]: all job posted by the company

        average_rating: float: average rating of the company

        num_rating: int: number of people who rated the company

        all_job: str: links to all the job posted by the company
    """

    name = 'companies_info'

    start_urls = []
    file_path = '/home/huynhhao/Desktop/job_recommender/crawl_data/companies_url.jl'
    with open(file_path, 'r') as f:
        for line in f:
            url = json.loads(line)['url']
            start_urls.append(url)

    def parse(self, response):
        
        company_name = response.url.split('/')[-1]
        company_info = {'company_name': company_name }

        company_info['name'] = response.xpath("//h1[@class='headers__info__name']/text()").get().strip()
        
        header_info = response.xpath("//div[@class='svg-icon__text']/text()").getall()
        
        company_info['city'] = header_info[0].strip() if header_info else None
        company_info['type'] = header_info[1].strip()  if 1 < len(header_info) else None
        company_info['num_employee'] = header_info[2].strip() if 2 < len(header_info) else None
        company_info['country'] = header_info[3].strip()  if 3 < len(header_info) else None
        company_info['working_day'] = header_info[4].strip() if 4 < len(header_info) else None
        company_info['OT'] = header_info[5].strip() if 5 < len(header_info) else None

        details = response.xpath("//div[@class='panel-paragraph']").getall()

        company_info['overview'] = details[0] if details else None
        company_info['expertise'] = details[1] if 1 < len(details) else None
        company_info['benifit'] = details[2] if 2 < len(details) else None

        company_info['average_rating'] = response.xpath("//span[@class='company-ratings__star-point']/text()").get()
        company_info['num_review'] = response.xpath("//li[@class='navigation__item review-tab']/a/text()").get()

        jobs = {}
        job_selectors = response.xpath("//div[@class='job']//h3[@class='title']/a")
        for selector in job_selectors:
            job_title = selector.xpath("text()").get()
            link = selector.xpath("@href").get()
            jobs[job_title] = response.urljoin(link)


        company_info['jobs'] = jobs
        
        company_info['logo'] = response.xpath("//div[@class='headers__logo__img']/picture/source/img/@data-src").get()

        yield company_info

        


                        
        