import numpy as np
import pandas as pd
import time
import math



sms_test_features = pd.read_csv("sms_test_features.csv")
sms_test_labels = pd.read_csv("sms_test_labels.csv")
sms_train_features = pd.read_csv("sms_train_features.csv")
sms_train_labels = pd.read_csv("sms_train_labels.csv")

sms_test_all = pd.merge(sms_test_features, sms_test_labels, on="Unnamed: 0")
sms_train_all = pd.merge(sms_train_features, sms_train_labels, on="Unnamed: 0")

# =================================
# 3.3
# =================================

# =================================
# Training
# =================================

start = time.perf_counter()


sms_train_spams = sms_train_all.loc[sms_train_all["class"] == 1]
sms_train_normal = sms_train_all.loc[sms_train_all["class"] == 0]

sms_train_spams_sum = sms_train_spams.sum()[1:]
sms_train_normal_sum = sms_train_normal.sum()[1:]

normal_count = sms_train_normal.shape[0]
spams_count = sms_train_spams_sum[-1]

"""
#add-one or Laplace smoothing

sms_train_spams_count = np.ones(len(sms_train_spams_sum), dtype="float64") + sms_train_spams_sum.astype("float64")
sms_train_normal_count = np.ones(len(sms_train_normal_sum), dtype="float64") + sms_train_normal_sum.astype("float64")
normal_count += len(sms_train_normal_sum)
spams_count += len(sms_train_spams_sum)
"""


p_spam = spams_count / (spams_count + normal_count)
p_normal = 1 - p_spam

total_spam_words = sum(sms_train_spams_sum[:-1])
p_word_given_spam = sms_train_spams_sum[:-1] / total_spam_words

total_normal_words = sum(sms_train_normal_sum[:-1])
p_word_given_normal = sms_train_normal_sum[:-1] / total_normal_words


finish = time.perf_counter()

print("The training computation time for Multiinomial Naive Bayes: " + str(finish - start) + "\n")


# =================================
# Testing
# =================================

def Arda_ln(array):

    result = np.zeros(len(array))

    for i in range(len(array)):

        if array[i] == 0:
            result[i] = -math.inf
        else:
            result[i] = np.log(array[i])

    return result

# we do this to not re compute the logs every time again and again
p_word_given_spam_np = np.array(p_word_given_spam)
p_word_given_spam_log = Arda_ln(p_word_given_spam_np)

p_word_given_normal_np = np.array(p_word_given_normal)
p_word_given_normal_log = Arda_ln(p_word_given_normal_np)

p_spam_log = np.log(p_spam)
p_normal_log = np.log(p_normal)

def Multiinomial_naive_bayes_for_one(point):

    p_spam_for_point = p_spam_log

    for i in range(len(point)):

        if(math.isinf(p_word_given_spam_log[i]) and point[i] > 0): # if there is one negatif infinite in the sum we dont need to do the rest

            p_spam_for_point = -999999999
            break
        elif (not math.isinf(p_word_given_spam_log[i])):

            p_spam_for_point = point[i] * p_word_given_spam_log[i]



    p_normal_for_point = p_normal_log

    for i in range(len(point)):

        if math.isinf(p_word_given_normal_log[i]) and point[i] > 0:  # if there is one negatif infinite in the sum we dont need to do the rest

            p_normal_for_point = -999999999
            break

        elif (not math.isinf(p_word_given_normal_log[i])):

            p_normal_for_point = point[i] * p_word_given_normal_log[i]


    if p_spam_for_point > p_normal_for_point:  # in equality we chose not spam

        return 1

    else:

        return 0


def Multiinomial_naive_bayes_for_set(set):
    predictions = np.zeros(set.shape[0])

    for i in range(set.shape[0]):
        temp = Multiinomial_naive_bayes_for_one(set[i, :])

        predictions[i] = temp

    return predictions


def Multiinomial_naive_bayes_results(test):

    predic = Multiinomial_naive_bayes_for_set(test[:,:-1])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(test.shape[0]):

        if test[i, -1] == 1:

            if predic[i] == 1:

                TP += 1

            else:

                FN += 1

        else:

            if predic[i] == 0:

                TN += 1

            else:

                FP += 1



    return (TP + TN) / (TP + TN + FP + FN), TP, TN, FP, FN


sms_test_all_np = sms_test_all.to_numpy()[:, 1:]

start = time.perf_counter()
sucses, tp, tn, fp, fn = Multiinomial_naive_bayes_results(sms_test_all_np)
finish = time.perf_counter()
print("The success for Multiinomial Naive Bayes: " + str(sucses))
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("with test computation time: " + str(finish - start) + "\n")








# =================================
# 3.3
# =================================


sms_test_all = pd.merge(sms_test_features, sms_test_labels, on="Unnamed: 0")
sms_train_all = pd.merge(sms_train_features, sms_train_labels, on="Unnamed: 0")

# =================================
# Training
# =================================

start = time.perf_counter()


sms_train_spams = sms_train_all.loc[sms_train_all["class"] == 1]
sms_train_normal = sms_train_all.loc[sms_train_all["class"] == 0]

sms_train_spams_count = np.count_nonzero(sms_train_spams, axis=0)[1:]
sms_train_normal_count = np.count_nonzero(sms_train_normal, axis=0)[1:]

normal_count = sms_train_normal.shape[0]
spams_count = sms_train_spams_count[-1]

#add-one or Laplace smoothing

sms_train_spams_count = np.ones(len(sms_train_spams_count), dtype="float64") + sms_train_spams_count.astype("float64")
sms_train_normal_count = np.ones(len(sms_train_normal_count), dtype="float64") + sms_train_normal_count.astype("float64")
normal_count += 2
spams_count += 2

p_spam = spams_count / (spams_count + normal_count)
p_normal = 1 - p_spam


p_word_given_spam = sms_train_spams_count[:-1] / spams_count
p_word_given_normal = sms_train_normal_count[:-1] / normal_count


finish = time.perf_counter()

print("The training computation time for Bernoulli Naive Bayes: " + str(finish - start) + "\n")



# =================================
# Testing
# =================================

def Arda_ln(array): # log function which does not spend to much time computing the log(0)

    result = np.zeros(len(array))

    for i in range(len(array)):

        if array[i] == 0:
            result[i] = -math.inf
        else:
            result[i] = np.log(array[i])

    return result

# we do this to not re compute the logs every time again and again
p_word_given_spam_np = np.array(p_word_given_spam)
p_word_given_spam_log = Arda_ln(p_word_given_spam_np)
p_word_given_spam_minus_one_log = np.log(np.ones(len(p_word_given_spam_np), dtype="float64") - p_word_given_spam_np)

p_word_given_normal_np = np.array(p_word_given_normal)
p_word_given_normal_log = Arda_ln(p_word_given_normal_np)
p_word_given_normal_minus_one_log = np.log(np.ones(len(p_word_given_normal_np), dtype="float64") - p_word_given_normal_np)

p_spam_log = np.log(p_spam)
p_normal_log = np.log(p_normal)

def Bernoulli_naive_bayes_for_one(point):

    p_spam_for_point = p_spam_log

    for i in range(len(point)):

        if point[i] > 0: # if there is one negatif infinite in the sum we dont need to do the rest

            p_spam_for_point += p_word_given_spam_log[i]

        else:

            p_spam_for_point += p_word_given_spam_minus_one_log[i]




    p_normal_for_point = p_normal_log

    for i in range(len(point)):

        if point[i] > 0:  # if there is one negatif infinite in the sum we dont need to do the rest

            p_normal_for_point += p_word_given_normal_log[i]

        else:

            p_normal_for_point += p_word_given_normal_minus_one_log[i]

    if p_spam_for_point > p_normal_for_point:  # in equality we chose not spam

        return 1

    else:

        return 0


def Bernoulli_naive_bayes_for_set(set):
    predictions = np.zeros(set.shape[0])

    for i in range(set.shape[0]):
        temp = Bernoulli_naive_bayes_for_one(set[i, :])

        predictions[i] = temp

    return predictions


def Bernoulli_naive_bayes_results(test):

    predic = Bernoulli_naive_bayes_for_set(test[:,:-1])

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(test.shape[0]):

        if test[i, -1] == 1:

            if predic[i] == 1:

                TP += 1

            else:

                FN += 1

        else:

            if predic[i] == 0:

                TN += 1

            else:

                FP += 1



    return (TP + TN) / (TP + TN + FP + FN), TP, TN, FP, FN


sms_test_all_np = sms_test_all.to_numpy()[:, 1:]

start = time.perf_counter()
sucses, tp, tn, fp, fn = Bernoulli_naive_bayes_results(sms_test_all_np)
finish = time.perf_counter()
print("Bernoulli Naive Bayes success for all futures: " + str(sucses))
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("with test computation time: " + str(finish - start) + "\n")


sms_train_spams_count = sms_train_spams_count[:-1]
sms_train_normal_count = sms_train_normal_count[:-1]

number_contains_word_in_spam = sms_train_spams_count # N_11
number_not_contains_word_in_spam = np.ones(len(sms_train_spams_count)) * spams_count - sms_train_spams_count #N_01

number_contains_word_in_normal = sms_train_normal_count #N_10
number_not_contains_word_in_normal = np.ones(len(sms_train_spams_count)) * normal_count - sms_train_normal_count # N_00

mail_count = spams_count + normal_count

number_contains_word = number_contains_word_in_spam + number_not_contains_word_in_spam # N_1.
number_not_contains = mail_count - number_contains_word # N_0.

# spams_count N_.1
# normal_count N_.0



"""
N = mail_count
N_o1 = spams_count
N_o0 = normal_count

mutial_information = []

for word in range(len(number_not_contains)):

    N_11 = number_contains_word_in_spam[word]
    N_01 = number_not_contains_word_in_spam[word]

    N_10 = number_contains_word_in_normal[word]
    N_00 = number_not_contains_word_in_normal[word]

    N_1o = N_11 + N_10
    N_0o = N_01 + N_00

    top = [N_11, N_01, N_10, N_00]
    bottom = [N_o1 * N_1o, N_0o * N_o1, N_1o * N_o0, N_0o * N_o0]

    mi = 0
    
    for i in range(4):
    
        mi += (top[i] / mail_count) * math.log((mail_count * top[i]) / bottom[i])

    mutial_information.append(mi)


mutial_information = np.array(mutial_information)
"""

mutial_information = number_contains_word_in_spam/mail_count * np.log((mail_count * number_contains_word_in_spam) / (number_contains_word * spams_count)) + (
    number_not_contains_word_in_spam/mail_count * np.log((mail_count * number_not_contains_word_in_spam) / (number_not_contains * spams_count))) + (
    number_contains_word_in_normal/mail_count * np.log((mail_count * number_contains_word_in_normal) / (number_contains_word * normal_count))) + (
    number_not_contains_word_in_normal/mail_count * np.log((mail_count * number_not_contains_word_in_normal) / (number_not_contains * normal_count)))

future_importance_order = np.argsort(mutial_information)
future_importance_order2 = np.append(future_importance_order, -1)

#these will be re-used so we need to save them somewhere
safe_p_word_given_spam_log = np.copy(p_word_given_spam_log)
safe_p_word_given_spam_minus_one_log = np.copy(p_word_given_spam_minus_one_log)

safe_p_word_given_normal_log = np.copy(p_word_given_normal_log)
safe_p_word_given_normal_minus_one_log = np.copy(p_word_given_normal_minus_one_log)


#testing for different future numbers

for k in (100, 200, 300, 400, 500, 600):

    p_word_given_spam_log = []
    p_word_given_spam_minus_one_log = []

    p_word_given_normal_log = []
    p_word_given_normal_minus_one_log = []

    for i in range(k):

        p_word_given_spam_log.append(safe_p_word_given_spam_log[(i-100)])
        p_word_given_spam_minus_one_log.append(safe_p_word_given_spam_minus_one_log[(i-100)])

        p_word_given_normal_log.append(safe_p_word_given_normal_log[(i-100)])
        p_word_given_normal_minus_one_log.append(safe_p_word_given_normal_minus_one_log[(i-100)])



    sms_test_all_np = sms_test_all.to_numpy()[:, 1:]
    sms_test_all_np = sms_test_all_np[:,future_importance_order2[-(k+1):]] # +1 is the labbels

    start = time.perf_counter()
    sucses, tp, tn, fp, fn = Bernoulli_naive_bayes_results(sms_test_all_np)
    finish = time.perf_counter()
    print("Bernoulli Naive Bayes Success for " + str(k ) + " futures: " + str(sucses))
    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("with test computation time: " + str(finish - start) + "\n")


print("I am happy")
