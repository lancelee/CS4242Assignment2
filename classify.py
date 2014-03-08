#!/usr/bin/env python
# import codecs
import string
import json
import re
from time import time

# import cld
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import metrics
# from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC

def main():
    # labels for training classifier later
    orgs = ['apple', 'google', 'microsoft', 'twitter']

    label_train = []
    # open and extract labels from label_train.txt
    with open('data/label_train.txt') as file:
        for line in file:
            list = line.split(',')
            temp = []
            for entry in list:
                temp.append(entry.translate(string.maketrans("",""), string.punctuation))
            label_train.append(temp)
    print len(label_train)        

    tweets_train = []
    app_train = [[] for i in xrange(3)]
    goo_train = [[] for i in xrange(3)]
    mic_train = [[] for i in xrange(3)]
    twi_train = [[] for i in xrange(3)]
    # open and extract texts from tweets_train.txt
    with open('data/tweets_train.txt') as file:
        contents = file.readlines()
    print len(contents)
    for i in range(len(label_train)):
        text = process_line(contents[i])
        if label_train[0] == 'apple':
            if label_train[1] == 'positive':
                app_train[0].append(text)
            elif label_train[1] == 'negative':
                app_train[1].append(text)
            elif label_train[1] == 'neutral':
                app_train[2].append(text)    
        elif label_train[0] == 'google':
            if label_train[1] == 'positive':
                goo_train[0].append(text)
            elif label_train[1] == 'negative':
                goo_train[1].append(text)
            elif label_train[1] == 'neutral':
                goo_train[2].append(text)              
        elif label_train[0] == 'microsoft':
            if label_train[1] == 'positive':
                mic_train[0].append(text)
            elif label_train[1] == 'negative':
                mic_train[1].append(text)
            elif label_train[1] == 'neutral':
                mic_train[2].append(text)  
        elif label_train[0] == 'twitter':
            if label_train[1] == 'positive':
                twi_train[0].append(text)
            elif label_train[1] == 'negative':
                twi_train[1].append(text)
            elif label_train[1] == 'neutral':
                twi_train[2].append(text)  

def process_line(line):
    # json_dict = json.loads(line)
    text =  json.loads(line)['text']
    
    # bonus = grab_info(json_dict)
    # if "retweeted_status" in json_dict:
    #     bonus2 = grab_info(json_dict["retweeted_status"])
    #     bonus += bonus2

    # text += bonus
    # text += ' &' + language
    return text

main()










