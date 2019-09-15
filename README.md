Predicting-Company-In-News
==========================

Predicting which company is talked about in a news article. In this, using Flair (Model for Named Entity Recognition http://alanakbik.github.io/papers/coling2018.pdf) as predictor model for company names in the articles text. Dataset of news articles was scraped from http://www.goodnewsfinland.com.

Pretrained Flair model used, was trained over the English CoNLL-03 task and can recognize 4 different entity types - Organization being one of them. In this project, we utilized the organization predictions.

Dataset consisted of 992 news articles (all the articles found from goodnewsfinland). The predictions are in file: "company_predictions.csv". Python code for making the predictions is in file: "NER_on_goodnews.py".
The Scraped information (containing the text of article) is in folder: "news_scraper" and file is named "goodnews.csv". The scraper built is located at path "/news_scraper/news_scraper/spiders/news_spider.py".
