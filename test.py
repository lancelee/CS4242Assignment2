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

    def add_entry(self, pos_tagged_sentences, label, org_name):
        for sentence in pos_tagged_sentences:
            for tuple in sentence:
                word_type = tuple[2][0]
                # if the word type is adjectives, add to the hashtable with the label
                if word_type == 'JJ' or word_type == 'JJR' or word_type == 'JJS':
                    word = tuple[0]
                    if word != org_name:
                        if word not in self.dict:
                            self.dict[word] = [label]

    def get_dictionary(self):
        return self.dict                   


class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):

    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):

    def __init__(self, dictionary, dictionary_paths):
        files = [open(path, 'r') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = dictionary
        self.max_key_size = len(dictionary)
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

    def get_dict(self):
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
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])

def process_line(line):
    # json_dict = json.loads(line)
    try:
        text =  json.loads(line)['text']
    except:
        return ""
    # remove numbers and lines
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    return text.lower()

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
    

    # train the sentiment lexicon for all organizations
    for i in range(len(orgs)):
        dictionary = Dictionary()
        for j in range(len(tweets_train[i])):
            if label_train[i][j] != "neutral":
                splitted_sentences = splitter.split(tweets_train[i][j])
                pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
                dictionary.add_entry(pos_tagged_sentences, label_train[i][j], orgs[i]) 
        
        # let dictionaryTagger take in a dictionary, inc.yml, dec.yml and inv.yml only
        dicttagger = DictionaryTagger(dictionary.get_dictionary(), ['dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'])

        print("analyzing sentiment...")

        scores = []
        for j in range(len(tweets_test[i])):
            splitted_sentences = splitter.split(tweets_test[i][j])
            pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
            dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
            
            scores.append(sentiment_score(dict_tagged_sentences))
            
        count = 0    
        for j in range(len(scores)):
            if get_sentiment(scores[j]) == label_test[i][j]:
                count += 1
        accuracy = count / (len(scores) + 0.)
        print "Accuracy for " + orgs[i] + ": " + str(accuracy)         

