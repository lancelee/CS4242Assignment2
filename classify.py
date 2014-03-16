#!/usr/bin/env python
# import codecs
import string
import json
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


# loading stopwords
stopwordlist = []
with open('stopwordlist.txt') as f:
    contents = f.readlines()
    for word in contents:
        stopwordlist.append(word.rstrip())

regex_tok = nltk.tokenize.RegexpTokenizer(r'\w+')
with open('positive_word.txt') as file: positive_words = set([word.rstrip() for word in file])
with open('negative_word.txt') as file: negative_words = set([word.rstrip() for word in file])

def process_line(line):
    # json_dict = json.loads(line)
    # try:
    #     text = json.loads(line)['text']
    # except:
    #     return ""
    line = unicode(line, 'iso-8859-1') # IMPT!! UnicodeDecodingError will appear if this is not here
    text = json.loads(line)['text']

    #More modifications
    text = text.lower()
    tokens = regex_tok.tokenize(text)
    #tokens = [v[0] for v in filter(lambda x: x[1][0] in "JRV", tokens)]
    for word in tokens:
        if (word in positive_words): tokens.append("@POS")
        if (word in negative_words): tokens.append("@NEG")

    for i, word in enumerate(tokens):
        if (i == len(tokens) - 1) : break
        next_word = tokens[i+1]

        if (word == "n't" or word == "not"):
            tokens[i+1] = "NOT_"+next_word

    text = " ".join(tokens)

    # bonus = grab_info(json_dict)
    # if "retweeted_status" in json_dict:
    #     bonus2 = grab_info(json_dict["retweeted_status"])
    #     bonus += bonus2

    # text += bonus
    # text += ' &' + language

    # text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    # text = text.lower()
    # list_of_words = text.split()
    # list_of_tuples = do_POS_tagging(list_of_words)
    # # only return words that are adjectives
    # adjectives = []
    # for tuple in list_of_tuples:
    #     if tuple[1] == 'JJ' or tuple[1] == 'JJR' or tuple[1] == 'JJS':
    #         adjectives.append(tuple[0])
    # return ' '.join(adjectives)
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


pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, lowercase=False, stop_words=stopwordlist)),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
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

    # print out the accuracy
    count = 0
    for i in range(len(orgs)):
        for j in range(len(predicted[i])):
            if predicted[i][j] == label_test[i][j]:
                count += 1
    accuracy = count / 928.0
    print accuracy


if __name__ == "__main__":
    main()
