import scrapy
import csv
import os.path


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ['goodnewsfinland.com']
    
    def start_requests(self):
        urls = [f"http://www.goodnewsfinland.com/feature/page/{x+1}/" for x in range(91)]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        links = response.xpath('//*[@class="pikku-uutinen"]//*[contains(@class, "pikku-uutinen-content")]/a/@href').extract()
        for link in links:
            yield scrapy.Request(link , callback=self.parse_article)
    
    def parse_article(self, response):
        # extracting articles header
        header = response.xpath('//*[@class="content"]/article/h1/text()').extract()
        header = header[0] # getting the string out of the list
        
        # extracting articles content
        paragraphs = response.xpath('//*[@class="content"]/article//p/text()').extract()

        # 1. removing first three paragraphs, that are not part of content, we want to extract
        paragraphs = [p for p in  paragraphs if p not in paragraphs[0:3]]
        # 2. concatenating the paragraphs to one text
        paragraphs = " ".join(paragraphs)
        # 3. replacing " " with " ", also replacing " ." with "." and " ," with ","
        paragraphs.replace("  ", " ")
        paragraphs.replace(" .", ".")
        paragraphs.replace(" ,", ",")

        # stripping the paragraph
        paragraphs = paragraphs.strip()

        # Removing "Share" from start if present (is not part of content)
        if paragraphs[0:5] == "Share":
            paragraphs = paragraphs[6:]

        
        # saving to csv file named:
        filename = 'goodnews.csv'

        # checking if csv file exists, if not, create one and make header row
        if not os.path.isfile(filename):
            header_row = ['link', 'header', 'content']
            with open(filename, 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(header_row)   

        # saving the extracted data from article as row in csv file
        row = [response.url, header, paragraphs]
        with open(filename, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        self.log(f"appended file {filename}")
         