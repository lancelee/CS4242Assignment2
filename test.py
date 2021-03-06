from pprint import pprint
import nltk
import yaml
import sys
import os
import re
import json
import string
from sklearn import metrics
import operator

def value_of(sentiment):
    if sentiment == 'positive': 
        return 1
    elif sentiment == 'very positive':
        return 3
    elif sentiment == 'negative': 
        return -1
    elif sentiment == 'very negative':
        return -3
    else: 
        return 0

def sentiment_score(list_of_tuples):
    total_score = 0
    
    for tuple in list_of_tuples:
        total_score += value_of(tuple[1])
    # normalize the scores
    return total_score 

def get_sentiment(score):
    if score < -3:
        return 'negative'
    elif score > 3:
        return 'positive'
    else:
        return 'neutral'

def process_line(line):
    # json_dict = json.loads(line)
    line = unicode(line, 'iso-8859-1') # IMPT!! UnicodeDecodingError will appear if this is not here
    text =  json.loads(line)['text']

    # remove numbers and lines
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = text.lower()
    
    # remove stopwords and stem
    global stemmer
    filtered_words = [stemmer(w) for w in text.split() if not w in nltk.corpus.stopwords.words('english')]
    text = ' '.join(filtered_words)
    return text

if __name__ == "__main__":
    orgs = ['apple', 'google', 'microsoft', 'twitter']
    label = ['positive', 'negative', 'neutral']

    stemmer = nltk.PorterStemmer().stem

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
			# add to dictionary
			dictionary[stemmer(word)] = polarity

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


    for i in range(len(orgs)):
        dic = {}
        for j in range(len(tweets_train[i])):
            list_of_words = tweets_train[i][j].split()
            for word in list_of_words:
                # add to dic
                if word not in dic:
                    dic[word] = 1
                else:
                    freq = dic[word]
                    dic[word] = freq + 1
        sorted_dic = sorted(dic.iteritems(), key=operator.itemgetter(1))
        print sorted_dic[-10:]


    # #
    # #   Initializing testing data
    # #
    # label_test = [[] for i in xrange(4)]
    # # open and extract labels from label_train.txt
    # with open('data/label_test.txt') as file:
    #     for line in file:
    #         list = line.split(',')
    #         name = list[0].translate(string.maketrans("",""), string.punctuation)  
    #         if name == 'apple':
    #             label_test[0].append(list[1].translate(string.maketrans("",""), string.punctuation))
    #         elif name == 'google':
    #             label_test[1].append(list[1].translate(string.maketrans("",""), string.punctuation))
    #         elif name == 'microsoft':
    #             label_test[2].append(list[1].translate(string.maketrans("",""), string.punctuation))
    #         elif name == 'twitter':
    #             label_test[3].append(list[1].translate(string.maketrans("",""), string.punctuation))

    # APPLE_TEST = len(label_test[0])
    # GOOGLE_TEST = APPLE_TEST + len(label_test[1])
    # MICROSOFT_TEST = GOOGLE_TEST + len(label_test[2])
    # TWITTER_TEST = MICROSOFT_TEST + len(label_test[3])

    # tweets_test = [[] for i in xrange(4)]
    # # open and extract texts from tweets_test.txt
    # with open('data/tweets_test.txt') as file:
    #     contents = file.readlines()
    # for i in range(len(contents[:APPLE_TEST])):
    #     text = process_line(contents[i])
    #     tweets_test[0].append(text)
    # for i in range(len(contents[APPLE_TEST:GOOGLE_TEST])):
    #     text = process_line(contents[APPLE_TEST+i])
    #     tweets_test[1].append(text)
    # for i in range(len(contents[GOOGLE_TEST:MICROSOFT_TEST])):
    #     text = process_line(contents[GOOGLE_TEST+i])
    #     tweets_test[2].append(text)
    # for i in range(len(contents[MICROSOFT_TEST:])):
    #     text = process_line(contents[MICROSOFT_TEST+i])
    #     tweets_test[3].append(text)    


    # predicted = [[] for i in xrange(4)]  
    # for i in range(len(orgs)):	
    # 	print("analyzing sentiment...")

    #     # using the sentiment lexicons to evaluate the test data
        
    #     for j in range(len(tweets_test[i])):
    #         # split texts into words
    #         list_of_words = tweets_test[i][j].split()
    #         # tag them with POS tags
    #         pos_tagged_words = nltk.pos_tag(list_of_words)
    #         # tag them with words if found in the dictionary
    #         dict_tagged_words = []
    #         for tuple in pos_tagged_words:
    #             if tuple[0] in dictionary:
    #                 word, tag = tuple
    #                 tuple = (word, dictionary[word]) 
    #                 dict_tagged_words.append(tuple)                    
    #         # count the score from the list of words
    #         score = sentiment_score(dict_tagged_words)
    #         predicted[i].append(get_sentiment(score))
    
    #     # print out confusion matrix
    #     print(metrics.classification_report(label_test[i], predicted[i]))
    #     conf_matrix = metrics.confusion_matrix(label_test[i], predicted[i])
    #     print(conf_matrix)

    #     # matching test results with the groundtruths
    #     count = 0    
    #     for j in range(len(predicted[i])):
    #         if predicted[i][j] == label_test[i][j]:
    #             count += 1
    #     accuracy = count / (len(predicted[i]) + 0.)
    #     print "Accuracy for " + orgs[i] + ": " + str(accuracy)         
