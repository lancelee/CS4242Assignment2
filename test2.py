from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import json
import string


def value_of(sentiment):
    if sentiment == 'positive': 
        return 1
    elif sentiment == 'negative': 
        return -1
    else: 
        return 0
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
    total_score = sentence_score(review, None, 0.0)
    # normalize the scores
    if len(review) != 0:
        total_score /= len(review)
    return total_score 

def process_line(line):
    # json_dict = json.loads(line)
    line = unicode(line, 'iso-8859-1') # IMPT!! UnicodeDecodingError will appear if this is not here
    text =  json.loads(line)['text']

    # remove numbers and lines
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = text.lower()
    
    # remove stopwords
    filtered_words = [w for w in text.split() if not w in nltk.corpus.stopwords.words('english')]
    text = ' '.join(filtered_words)
    return text

if __name__ == "__main__":
    orgs = ['apple', 'google', 'microsoft', 'twitter']
    label = ['positive', 'negative', 'neutral']


    #
    #   Initializing sentiment lexicon from sentiment_lexicon.tff
    #
    dictionary = {}
    with open('dicts/sentiment_lexicon.tff') as file:
		for line in file:
			wstrength, wlen, word, wpos, wstem, wpolarity = line.split()
			name, strength = wstrength.split("=")
			name, word = word.split("=")
			name, polarity = wpolarity.split("=")
			if strength == 'strongsubj':
				polarity = "very " + polarity
			print word + ": " + polarity
			# add to dictionary
			dictionary[word] = polarity

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

    
    print dictionary.items()

    # for i in range(len(orgs)):	
    # 	print("analyzing sentiment...")

        # # using the sentiment lexicons to evaluate the test data
        # scores = []
        # for j in range(len(tweets_test[i])):
        #     list_of_words = splitter.split(tweets_test[i][j])
        #     pos_tagged_words = postagger.pos_tag(list_of_words)
        #     dict_tagged_words = dicttagger.tag(pos_tagged_words)
        #     score = sentiment_score(dict_tagged_words)
        #     print score
        #     scores.append(score)
    
        # # matching test results with the groundtruths
        # count = 0    
        # for j in range(len(scores)):
        #     if get_sentiment(scores[j]) == label_test[i][j]:
        #         count += 1
        # accuracy = count / (len(scores) + 0.)
        # print "Accuracy for " + orgs[i] + ": " + str(accuracy)         
