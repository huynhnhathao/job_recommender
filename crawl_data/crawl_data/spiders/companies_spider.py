import scrapy


class CompaniesSpider(scrapy.Spider):
    """This spider wil crawl all the company link available in itviec and save it
    to a json line file.
    """
    name = "Companies"
    start_urls = [
        'https://itviec.com/companies',
    ]

    def parse(self, response):
        all_companies = response.xpath("//div[@class='first-group companies']/a[@class='featured-company']/@href").getall()
        for company_link in all_companies:
            relative_link = '/'.join(company_link.split('/') [:-1])
            company_name = company_link.split('/') [-2]
            absolute_link = response.urljoin(relative_link)
            yield {company_name: absolute_link }

        next_page = response.xpath("//a[@class='more-jobs-link more-company']/@href").get()
        # next_page now has the form of '/companies?page=2' or None

        if next_page is not None:
            # makes absolute url
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback = self.parse)

