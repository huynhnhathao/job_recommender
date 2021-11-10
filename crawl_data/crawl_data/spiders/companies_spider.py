import scrapy


class CompaniesSpider(scrapy.Spider):
    name = "Compannies"
    start_urls = [
        'https://itviec.com/companies',
    ]

    def parse(self, response):
        all_companies = response.xpath("//div[@class='first-group companies']/a[@class='featured-company']/@href").getall()
        yield all_companies

        next_page = response.xpath("//a[@class='more-jobs-link more-company']/@href").get()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback = self.parse)

