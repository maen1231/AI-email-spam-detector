import os
import math
import numpy as np


most_common_word = 3000
#avoid 0 terms in features
smooth_alpha = 1.0;
class_num =2;#we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]#probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word))#feature parameterized probability
SPAM = 1#spam class label
HAM = 0#ham class label


class MultinomialNB_class:
    
    #multinomial naive bayes
    def MultinomialNB(self, features, labels):
        label_count = np.zeros(2)
        for i in labels :
            label_count[int(i)] += 1

        class_log_prior[0] = math.log(float(label_count[0])/float(len(labels)))#ham
        class_log_prior[1] = math.log(float(label_count[1])/float(len(labels)))#spam

        #calculate feature_log_prob
        ham = np.zeros(most_common_word)
        spam = np.zeros(most_common_word)
        sum_ham = 0
        sum_spam = 0

        for j in range(len(features)) :
            for k in range(len(features[j])) :
                if j < label_count[0] :
                    ham[k] += features[j][k]
                    sum_ham += features[j][k]
                else :
                    spam[k] += features[j][k]
                    sum_spam += features[j][k]

        for l in range(most_common_word) :
            ham[l] += smooth_alpha
            spam[l] += smooth_alpha

        sum_ham += smooth_alpha*most_common_word
        sum_spam += smooth_alpha*most_common_word

        for h in range(most_common_word) :
            feature_log_prob[0][h] = math.log(float(ham[h])/float(sum_ham))
            feature_log_prob[1][h] = math.log(float(spam[h])/float(sum_spam))

    #multinomial naive bayes prediction
    def MultinomialNB_predict(self, features):
        classes = np.zeros(len(features));

        ham_prob = 0.0;
        spam_prob = 0.0;
        for i in range(len(features)) :
            ham_prob = 0.0;
            spam_prob = 0.0;
            for j in range(len(features[i])) :
                ham_prob += feature_log_prob[0][j]*float(features[i][j])
                spam_prob += feature_log_prob[1][j]*float(features[i][j])

            ham_prob += class_log_prior[0]
            spam_prob += class_log_prior[1]

            if ham_prob > spam_prob :
                classes[i] = HAM
            else :
                classes[i] = SPAM

        return classes
