from csv import DictReader
from operator import contains
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from MultinomialNB import MultinomialNB_class
import nltk
import distutils
import pandas as pd
from distutils import dir_util

nltk.download(['stopwords', 'punkt'])
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('opinion_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import shutil

test = 'SpamEmails\\test-mails'
train = 'SpamEmails\\train-mails'

all_mails = '.SpamEmails\\all-mails\\'

test_file_path = 'SpamEmails\\test-mails_new'
train_file_path = 'SpamEmails\\train-mails_new'

if not os.path.exists(test_file_path):
    # Create a new directory because it does not exist
    os.makedirs(test_file_path)

if not os.path.exists(train_file_path):
    # Create a new directory because it does not exist
    os.makedirs(train_file_path)


def clear_content(folder):
    if len(os.listdir(folder)) > 0:
        for file in os.listdir(folder):
            file_path = folder + "\\" + file
            os.remove(file_path)


def split_data(parent_folder=all_mails, train_ratio=0.8):
    clear_content(test_file_path)
    clear_content(train_file_path)

    n_train = len(os.listdir(parent_folder)) * train_ratio
    n_each_mail = math.floor(n_train / 2)
    count_ham = 0
    count_spam = 0
    # move files to training folder
    for file_name in os.listdir(parent_folder):
        file_name = parent_folder + file_name
        if count_spam <= n_each_mail:
            if 'spm' in file_name:
                shutil.move(file_name, train_file_path)
                count_spam += 1

        if count_ham <= n_each_mail:
            if 'msg' in file_name:
                shutil.move(file_name, train_file_path)
                count_ham += 1

    # copy all other files to test folder
    distutils.dir_util.copy_tree(parent_folder, test_file_path)


def word_exclusion(text):
    ''' preprocessing function uses nltk to tokenize text for vectorizing
    parameter: cleaned dataframe
    return: dataframe with essential text in "Review text" column '''

    # filter stop words and punctuation
    def extract_words(text):
        stop_words = set(stopwords.words('english')) - {'not'}
        word_tokens = word_tokenize(text)
        filtered_text = [w.lower().strip() for w in word_tokens if
                         not (w.strip() == '' or w in stop_words or w in [',', '.', '|', '~']) and not (
                                 len(w) == 1 and isinstance(w, str))]
        return filtered_text

    def lemmatizer_on_text(text):
        # replaces misspelled word with closest approximation from the WordNet corpus
        lm = WordNetLemmatizer()
        lemmed_text = "".join(lm.lemmatize(w) + " " for w in text)
        return lemmed_text

    text = extract_words(text)
    text = lemmatizer_on_text(text)

    return text


def run_classifier(train_ratio=0.8):
    distutils.dir_util.copy_tree(src=train, dst=all_mails)
    distutils.dir_util.copy_tree(src=test, dst=all_mails)

    wordMap = {}
    commonMap = []

    most_common_word = 3000

    # avoid 0 terms in features
    smooth_alpha = 1.0

    class_num = 2  # we have only two classes: ham and spam
    class_log_prior = [0.0, 0.0]  # probability for two classes
    feature_log_prob = np.zeros((class_num, most_common_word))  # feature parameterized probability
    SPAM = 1  # spam class label
    HAM = 0  # ham class label

    # read file names in the specific file path
    def read_file_names(file_path):
        return os.listdir(file_path)

    # read in the specific file
    def read_file(file):
        content = ''
        with open(file) as f:
            for line in f:
                line = line.strip()
                if line:
                    content += line
        content = word_exclusion(content.strip())
        return content

    # count the total words
    def count_total_word(words):
        for word in words:
            if not word.isalpha():
                continue
            if not (word in wordMap.keys()):
                wordMap[word] = 1
            else:
                count = wordMap[word]
                wordMap[word] = count + 1

    # count the word in one file
    def count_word(words, singleWordMap={}):
        for word in words:
            if not word.isalpha():
                continue
            if not (word in singleWordMap.keys()):
                singleWordMap[word] = 1
            else:
                count = singleWordMap[word]
                singleWordMap[word] = count + 1

            # find the most common words in files store in commonMap

    def most_common():
        # sort the wordMap
        sort_wordMap = {k: v for k, v in sorted(wordMap.items(), key=lambda x: x[1], reverse=True)}

        # add the most common words into commonMap
        index = 0
        for key in sort_wordMap.keys():
            if index < most_common_word:
                commonMap.append(key)
            else:
                break
            index += 1

    # generate features according to commonMap
    def generate_feature(features, path, files):
        singleWordMap = {}
        file_index = 0
        for file in files:
            singleWordMap = {}
            content = read_file(path + '\\' + file)
            content.replace("\n", "")
            contents = content.split(" ")
            # print(contents)
            count_word(contents, singleWordMap)

            for key1 in singleWordMap.keys():
                common_index = 0
                for key2 in commonMap:
                    if key1 == key2:
                        features[file_index][common_index] = singleWordMap[key1]
                    common_index += 1
            file_index += 1

    print('Split the data into %s %% training and %s %% testing' % (train_ratio * 100, 100 - train_ratio * 100))
    split_data(train_ratio=train_ratio)
    # construct dictionary
    files = read_file_names(train_file_path)

    for i in range(len(files)):
        content = read_file(train_file_path + '\\' + files[i])
        content.replace("\n", "")
        # print(type(content))
        contents = content.split(" ")
        count_total_word(contents)

    print("The maximum of most_common can be: ", len(wordMap))

    most_common()

    # construct model
    # training feature matrix
    train_features = np.zeros((len(files), len(commonMap)))
    generate_feature(train_features, train_file_path, files)

    # training labels
    train_labels = np.zeros(len(files))
    for i in range(len(files) // 2, len(files)):
        train_labels[i] = 1

    # verify model
    # load test data
    files = read_file_names(test_file_path)
    # testing feature matrix
    test_features = np.zeros((len(files), len(commonMap)))
    generate_feature(test_features, test_file_path, files)
    print(test_features)
    # testing labels
    test_labels = np.zeros(len(files))
    for i in range(len(files) // 2, len(files)):
        test_labels[i] = 1

    # Multinomial Naive Bayes start
    # print(train_labels)
    # train model
    MultinomialNB = MultinomialNB_class()
    MultinomialNB.MultinomialNB(train_features, train_labels)
    # test model
    classes = MultinomialNB.MultinomialNB_predict(test_features)
    error = 0
    for i in range(len(files)):
        if test_labels[i] == classes[i]:
            error += 1
    accuracy = (float(error) / float(len(test_labels))) * 100
    print("Accuracy of Multinomial Naive Bayes: %.2f%%" % accuracy)
    print('*' * 100)
    return accuracy
    # Multinomial Naive Bayes end


if __name__ == '__main__':
    acc1 = run_classifier(train_ratio=0.6)
    acc2 = run_classifier(train_ratio=0.75)
    acc3 = run_classifier(train_ratio=0.8)
    acc4 = run_classifier(train_ratio=0.85)

    # plot accuracy
    test = np.array(['60% training - 40% testing', '75% training - 25% testing', '80% training - 20% testing',
                     '85% training - 15% testing'])
    accuracy = np.array([acc1, acc2, acc3, acc4])
    acc_df = pd.DataFrame(data=[accuracy], columns=test.tolist())
    print(acc_df)
    plt.ylim(min(accuracy)-.5, max(accuracy)-1)
    plt.bar(test, accuracy, width=0.25, color='sandybrown')
    plt.show()
