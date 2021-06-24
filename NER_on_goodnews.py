from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd 
import sys
import itertools
import operator
import os
import csv
import re
# # # Predicting what organization/company is talked about in a news article # # # 

# Loading previously scraped good news finland data. (Scraped every news article from the site)
# (see scraper in folder "news_scraper/news_scraper/spiders/news_spider.py")
news_data = pd.read_csv("./news_scraper/goodnews.csv")
# opening with panda for easy manipulation and exploration

# Setting up Named entity recognition model 

# loading the NER model, and using it in tagger
tagger = SequenceTagger.load('ner')

# creating function for finding the organizations from given sentence
def sentence_to_org(sentence):
    Threshold=0.81
    try:
        sentence_tokenized = Sentence(sentence)
        tagger.predict(sentence_tokenized)
        sentence_dict = sentence_tokenized.to_dict(tag_type='ner')
        
        org_names = []
        for entity in sentence_dict['entities']:
            scores=0.0
            #dict: {'text': 'NBA', 'start_pos': 43, 'end_pos': 46, 'labels': [ORG (0.9879)]}
            print(entity['labels'])
            #print(entity['labels'][0])
            ent=str(entity['labels'][0])
            #print(' word :'+str( entity['labels']))

            print(ent)
            if 'ORG' in ent:
                res=''.join( (re.findall(r'\(.*?\)',ent))[0] )
                #print((res))
                res=(res.split('('))[1]
                #print((res))
                res=(res.split(')'))[0]
                #print((res))
                scores=float( res)
                if (scores-Threshold)>0:
                    print('a ORG word :'+entity['text'])
                    org_names.append(entity['text'])
                else:
                    pass
#             if entity['type'] == 'ORG':
#                 print('find ORG word :'+entity['text'])
#                 org_names.append(entity['text'])
        return org_names
    except Exception as e: 
        print(e)
    
# helper function for finding the most common organization in the sentence
def most_common(L):
    if not L:
        return None
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

# helper function for checking if the header contains some of the companies found. 
def does_header_contain(header, companies):
    if not companies:
        return None
    
    for company in companies:
        if company in header:
            return company
    return None

# file for predictions
filename = "company_predictons.csv"

if not os.path.isfile(filename):
    header_row = ['link to article', 'header', 'found company tags', 'prediction']
    with open(filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header_row)   

# testing out the "Company predictor function" on the scraped data.
# looping over the data and saving to new file: "company_predictons.csv"
for index, row in news_data.iterrows():
    # check first 20
    #if index == 10:
    #    break
    try:
        print("Link to article: ", row['link'])
        print("Article header:  ", str( row['header']))
    except:
        pass
    # getting all the company tags from sentence:
    companies = sentence_to_org(row['content'])
    print("all the found company tags: ", companies)

    # getting the most common company tag
    most_common_company = most_common(companies)

    # checking what company the header contains
    company_in_header = does_header_contain(row['header'], companies)

    # if company in header matches the most common company, predicting that article is about that company
    if most_common_company == company_in_header:
        predicted_company = most_common_company
    elif company_in_header == None:
        predicted_company = most_common_company
    elif company_in_header:
        predicted_company = company_in_header
    else: 
        predicted_company = None
    
    if predicted_company != None:
        print("Article is predicted to be about following company: ", predicted_company)
    else: 
        print("Article was not about any particular company.")
    print("\n")

    # saving to csv file the predictions
    row = [row['link'], row['header'], companies, predicted_company]
    #add , encoding="utf-8"
    with open(filename, 'a', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    