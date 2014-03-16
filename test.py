# -*- coding: utf-8 -*-
"""
basic_sentiment_analysis
~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the code and examples described in 
http://fjavieralba.com/basic-sentiment-analysis-with-python.html

"""

from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import json
import string

class Dictionary(object):

    def __init__(self):
        self.dict = {}

    def add_entry(self, pos_tagged_words, label, org_name):
        for tuple in pos_tagged_words:
            word_type = tuple[1]
            word = tuple[0]
            if word != org_name:
                if word not in self.dict:
                    self.dict[word] = label

    def get_dictionary(self):
        return self.dict                   


class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a sentence 
        output format: a list of words.
        """
        list_of_words = self.nltk_tokenizer.tokenize(text) 
        return list_of_words


class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentence):
        """
        input format: a list of words
        """
        pos = nltk.pos_tag(sentence)
        # filter out words you duwan
        filtered = []
        for tuple in pos:
            if (tuple[1] == 'JJ' or tuple[1] == 'JJR' or tuple[1] == 'JJS' 
                or tuple[1] == 'RB' or tuple[1] == 'RBR' or tuple[1] == 'RBS'):
                filtered.append(tuple)

        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = dictionary
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key not in self.dictionary:
                    self.dictionary[key] = curr_dict[key][0]

    def tag(self, postagged_words):
        """
            replace postags with tags from dictionary
        """
        tag_sentence = []

        for tuple in postagged_words:
            word = tuple[0]
            if word in self.dictionary:
                taggings = self.dictionary[word]
                tagged_word = (word, taggings)
                tag_sentence.append(tagged_word)
        return tag_sentence

    def get_dictionary(self):
        return self.dictionary

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):    
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tag = current_token[1]
        token_score = value_of(tag)
        if previous_token is not None:
            previous_tag = previous_token[1]
            if 'inc' in previous_tag:
                token_score *= 2.0
            elif 'dec' in previous_tag:
                token_score /= 2.0
            elif 'inv' in previous_tag:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    total_score = 0.0
    return sentence_score(review, None, total_score)

def process_line(line):
    # json_dict = json.loads(line)
    line = unicode(line, 'iso-8859-1') # IMPT!! UnicodeDecodingError will appear if this is not here
    text =  json.loads(line)['text']

    # remove numbers and lines
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = text.lower()
    return text

def get_sentiment(score):
    if score < 0:
        return 'negative'
    elif score > 0:
        return 'positive'
    else:
        return 'neutral'

if __name__ == "__main__":
    orgs = ['apple', 'google', 'microsoft', 'twitter']
    label = ['positive', 'negative', 'neutral']

    #
    #   Initializing training data
    #
    label_train = [[] for i in xrange(4)]
    # open and extract labels from label_train.txt
    with open('data/label_train.txt') as file:
        for line in file:
            list = line.split(',')
            name = list[0].translate(string.maketrans("",""), string.punctuation)  
            if name == 'apple':
                label_train[0].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'google':
                label_train[1].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'microsoft':
                label_train[2].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'twitter':
                label_train[3].append(list[1].translate(string.maketrans("",""), string.punctuation))

    APPLE_TRAIN = len(label_train[0])
    GOOGLE_TRAIN = APPLE_TRAIN + len(label_train[1])
    MICROSOFT_TRAIN = GOOGLE_TRAIN + len(label_train[2])
    TWITTER_TRAIN = MICROSOFT_TRAIN + len(label_train[3])

    tweets_train = [[] for i in xrange(4)]
    # open and extract texts from tweets_train.txt
    with open('data/tweets_train.txt') as file:
        contents = file.readlines()
    for i in range(len(contents[:APPLE_TRAIN])):
        text = process_line(contents[i])
        tweets_train[0].append(text)
    for i in range(len(contents[APPLE_TRAIN:GOOGLE_TRAIN])):
        text = process_line(contents[APPLE_TRAIN+i])
        tweets_train[1].append(text)
    for i in range(len(contents[GOOGLE_TRAIN:MICROSOFT_TRAIN])):
        text = process_line(contents[GOOGLE_TRAIN+i])
        tweets_train[2].append(text)
    for i in range(len(contents[MICROSOFT_TRAIN:])):
        text = process_line(contents[MICROSOFT_TRAIN+i])
        tweets_train[3].append(text)    

    #
    #   Initializing testing data
    #
    label_test = [[] for i in xrange(4)]
    # open and extract labels from label_train.txt
    with open('data/label_test.txt') as file:
        for line in file:
            list = line.split(',')
            name = list[0].translate(string.maketrans("",""), string.punctuation)  
            if name == 'apple':
                label_test[0].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'google':
                label_test[1].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'microsoft':
                label_test[2].append(list[1].translate(string.maketrans("",""), string.punctuation))
            elif name == 'twitter':
                label_test[3].append(list[1].translate(string.maketrans("",""), string.punctuation))

    APPLE_TEST = len(label_test[0])
    GOOGLE_TEST = APPLE_TEST + len(label_test[1])
    MICROSOFT_TEST = GOOGLE_TEST + len(label_test[2])
    TWITTER_TEST = MICROSOFT_TEST + len(label_test[3])

    tweets_test = [[] for i in xrange(4)]
    # open and extract texts from tweets_test.txt
    with open('data/tweets_test.txt') as file:
        contents = file.readlines()
    for i in range(len(contents[:APPLE_TEST])):
        text = process_line(contents[i])
        tweets_test[0].append(text)
    for i in range(len(contents[APPLE_TEST:GOOGLE_TEST])):
        text = process_line(contents[APPLE_TEST+i])
        tweets_test[1].append(text)
    for i in range(len(contents[GOOGLE_TEST:MICROSOFT_TEST])):
        text = process_line(contents[GOOGLE_TEST+i])
        tweets_test[2].append(text)
    for i in range(len(contents[MICROSOFT_TEST:])):
        text = process_line(contents[MICROSOFT_TEST+i])
        tweets_test[3].append(text)    


    splitter = Splitter()
    postagger = POSTagger()
    
    # write to file the positive and negative texts for analysis
    positive_output = open("positive.txt", "wb")
    negative_output = open("negative.txt", "wb")
    neutral_output = open("neutral.txt", "wb")

    for i in range(len(tweets_train)):
        for j in range(len(tweets_train[i])):
            if label_train[i][j] == 'positive':
                positive_output.write(tweets_train[i][j].encode('utf8') + '\n')
            elif label_train[i][j] == 'negative':
                negative_output.write(tweets_train[i][j].encode('utf8') + '\n')
            else:    
                neutral_output.write(tweets_train[i][j].encode('utf8') + '\n')

    positive_output.close()
    negative_output.close()
    neutral_output.close()


    # train the sentiment lexicon for all organizations
    for i in range(len(orgs)):
        dictionary = Dictionary()
        for j in range(len(tweets_train[i])):
            if label_train[i][j] != "neutral":
                list_of_words = splitter.split(tweets_train[i][j])
                pos_tagged_words = postagger.pos_tag(list_of_words)
                dictionary.add_entry(pos_tagged_words, label_train[i][j], orgs[i]) 

        # let dictionaryTagger take in a dictionary, inc.yml, dec.yml and inv.yml only
        dicttagger = DictionaryTagger(dictionary.get_dictionary(), ['dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])
        print("analyzing sentiment...")
 

        scores = []
        for j in range(len(tweets_test[i])):
            list_of_words = splitter.split(tweets_test[i][j])
            pos_tagged_words = postagger.pos_tag(list_of_words)
            dict_tagged_words = dicttagger.tag(pos_tagged_words)
            score = sentiment_score(dict_tagged_words)
            print score
            scores.append(score)
    
        count = 0    
        for j in range(len(scores)):
            if get_sentiment(scores[j]) == label_test[i][j]:
                count += 1
        accuracy = count / (len(scores) + 0.)
        print "Accuracy for " + orgs[i] + ": " + str(accuracy)         

