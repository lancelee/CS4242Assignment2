#!/usr/bin/env python
# import codecs
import string
import json
import numpy
import re
from time import time
import sys
import nltk

# import cld
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob

stemmer = nltk.PorterStemmer().stem

stopwordlist = []
with open('stopwordlist.txt') as f:
    contents = f.readlines()
    for word in contents:
        stopwordlist.append(word.rstrip())

#
#   Initiating Bing Liu's sentiment lexicon 
#
regex_tok = nltk.tokenize.RegexpTokenizer(r'\w+')
with open('bing_positive.txt') as file: positive_words = set([stemmer(word.rstrip()) for word in file])
with open('bing_negative.txt') as file: negative_words = set([stemmer(word.rstrip()) for word in file])


#
# Initiating a list of common words
#
common_words = [u'app', u'ios5', u'get', u'rt', u'iphon', u'store', u'4s', u'siri', u'appl', u'ic', u'samsung', u'nexu', u'sandwich'
                    u'ice', u'cream', u'android', u'googl', u'nokia', u'cloud', u'via', u'steve', u'ballmer', u'phone', u'window'
                    u'microsoft', u'im', u'facebook', u'twitter']


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


def process_line(line, reprocess=False):
    # json_dict = json.loads(line)
    # try:
    #     text = json.loads(line)['text']
    # except:
    #     return ""
    line = unicode(line, 'iso-8859-1') # IMPT!! UnicodeDecodingError will appear if this is not here
    js = json.loads(line)
    text = js['text']

    
    # perform reprocess
    if reprocess == True:
        pass


    # blob = TextBlob(text)

    #More modifications
    text = text.lower()
    tokens = [stemmer(word) for word in regex_tok.tokenize(text)]
    #tokens = [v[0] for v in filter(lambda x: x[1][0] in "JRV", nltk.pos_tag(tokens))]

    # global common_words
    # tokens = [word for word in tokens if not word in common_words]
    
    global dictionary
    
    to_negate = False   
    for word in tokens:
    #     # handling negations        
    #     if word == 'but':
    #         if tokens[-1] == "@NEG" or tokens[-1] == "@POS": 
    #             tokens.pop()  # pop out the last sentiment tag if it's positve or negative
    #     if word == 'not':
    #         to_negate = True

        # # using Bing Liu's lexicon    
        # if (word in positive_words): tokens.append("@POS")
        # if (word in negative_words): tokens.append("@NEG")

        # using MPQA lexicon
        if word in dictionary:
            if dictionary[word] == 'positive':
                if to_negate == True:
                    tokens.append("@NEG")
                    to_negate = False
                else:
                    tokens.append("@POS")
            elif dictionary[word] == 'very positive':
                if to_negate == True:
                    tokens.append("@NEG")
                    tokens.append("@NEG")
                    to_negate = False
                else:
                    tokens.append("@POS")
                    tokens.append("@POS")
            elif dictionary[word] == 'negative':
                if to_negate == True:
                    tokens.append("@POS")
                    to_negate = False
                else:
                    tokens.append("@NEG")
            elif dictionary[word] == 'very negative':
                if to_negate == True:
                    tokens.append("@POS")
                    tokens.append("@POS")
                    to_negate = False
                else:
                    tokens.append("@NEG")
                    tokens.append("@NEG")
            elif dictionary[word] == 'neutral':
                if to_negate == True:
                    to_negate = False
                tokens.append("@NEU")
            elif dictionary[word] == 'very neutral':    
                if to_negate == True:
                    to_negate = False
                tokens.append("@NEU")
                tokens.append("@NEU")
    
    # tokens.append('@PATTERN_POLARITY' + str(round(blob.sentiment.polarity, 1)))

    text = " ".join(tokens)

    return text

def do_POS_tagging(list_of_words):
    list_of_tuples = nltk.pos_tag(list_of_words)
    return list_of_tuples


# labels for training classifier later
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

vectorizer = CountVectorizer(max_df=0.5, lowercase=False)
clf = LinearSVC()

pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', clf),
])

parameters = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'vect__stop_words': ('english', stopwordlist),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    # 'clf__class_weight': (None, 'auto'),
    # 'clf__multi_class': ('ovr', 'crammer_singer'),
    # 'clf__C': (1.0, 2.0, 3.0, 4.0, 5.0),
    # 'clf__loss': ('l1', 'l2'),
    # 'clf__penalty': ('l1', 'l2'),
}

GRID_SEARCH = False

predicted = [0]*4

def main():
    global predicted
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    for i in range(len(orgs)):
        classifier = pipeline

        if GRID_SEARCH:
            classifier = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

            print("Performing grid search for " + orgs[i] + " ...")
            # print("pipeline:", [name for name, _ in pipeline.steps])
            # print("parameters:")
            # pprint(parameters)
            t0 = time()
            # training phase
            classifier.fit(tweets_train[i], label_train[i])
            print("done in %0.3fs" % (time() - t0))
            print("\n")

            print("Best score: %0.3f" % classifier.best_score_)
            # print("Best parameters set:")
            # best_parameters = classifier.best_estimator_.get_params()
            # for param_name in sorted(parameters.keys()):
            #    print("\t%s: %r" % (param_name, best_parameters[param_name]))
        else:
            print("Training classifier for " + orgs[i] + " ...")
            classifier.fit(tweets_train[i], label_train[i])

        # testing classifer with testset and groundtruths
        print("Best score with test set: %0.3f" % classifier.score(tweets_test[i], label_test[i]))
        predicted[i] = classifier.predict(tweets_test[i])
        print(metrics.classification_report(label_test[i], predicted[i]))
        conf_matrix = metrics.confusion_matrix(label_test[i], predicted[i])
        print(conf_matrix)

        # with open(orgs[i] + "_wrong_prediction.txt", 'w') as wrong_file:
        #     for j in range(len(tweets_test[i])):
        #         if (label_test[i][j] != predicted[i][j]):
        #             line = '"' + tweets_test[i][j] + '"' + " classified as " + predicted[i][j] + " but should be " + label_test[i][j] + "\n\n"
        #             wrong_file.write(line.encode('UTF-8'))

        with open('dicts/' + orgs[i] + "_trained_lexicon.txt", 'w') as trained_lexicon:
            feature_names = numpy.asarray(vectorizer.get_feature_names())
            print("top 20 keywords per class:")
            for j, category in enumerate(['negative', 'neutral', 'positive']):
                top20 = numpy.argsort(clf.coef_[j])[-20:]
                print("%s: %s" % (category, " ".join(feature_names[top20])))
                trained_lexicon.write(category + ': ' + " ".join(feature_names[top20]) + "\n")
        trained_lexicon.close()
    # print out the accuracy
    count = 0
    for i in range(len(orgs)):
        for j in range(len(predicted[i])):
            if predicted[i][j] == label_test[i][j]:
                count += 1
    accuracy = count / 928.0
    print 'Total Accuracy: ' + str(accuracy)


if __name__ == "__main__":
    main()
