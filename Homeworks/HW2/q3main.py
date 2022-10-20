import numpy as np
import pandas as pd
import random

def min_max_normalize(data):
    min = np.min(data)
    max = np.max(data)
    delta = max-min

    result = (data - min) / delta

    return result

def confusion_matrix_maker(prediction, labels):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(labels.shape[0]):

        if labels[i] == 1:

            if prediction[i] == 1:

                TP += 1

            else:

                FN += 1

        else:

            if prediction[i] == 0:

                TN += 1

            else:

                FP += 1

    return TP, TN, FP, FN

def Question3():


    features_train = pd.read_csv("question-3-features-train.csv")
    features_test = pd.read_csv("question-3-features-test.csv")

    labels_train = pd.read_csv("question-3-labels-train.csv")
    labels_test = pd.read_csv("question-3-labels-test.csv")

    # we need to do the min max normilization to all of the data at the same time

    features = features_train.append(features_test)
    labels = labels_train.append(labels_test)

    features = features.to_numpy()
    labels = labels.to_numpy()

    temp_features = np.ones((891,4))

    temp_features[:, 1] = min_max_normalize(features[:, 0])
    temp_features[:, 2] = min_max_normalize(features[:, 1])
    temp_features[:, 3] = min_max_normalize(features[:, 2])


    features_train_array = temp_features[:712, :]
    features_test_array = temp_features[712:, :]

    labels_train_array = labels[:712, :]
    labels_test_array = labels[712:, :]


    n, m = features_train_array.shape



    print("\n***** Question 3.1: *****\n")



    learn_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]


    for learn_rate in learn_rates:

        weights = np.zeros((4, 1))

        for _ in range(1000):

            old_weights = weights

            """
            aa1 = np.exp(np.dot(features_train_array, old_weights))
        
            aa11 = (aa1 / (1 + aa1))
        
            aa2 = np.subtract(labels_train_array, aa11)
        
            aa3 = np.dot(features_train_array[:, :].T, aa2)
        
            aa4 = learn_rate * aa3
        
            weights = old_weights + aa4
            """

            weights = old_weights + learn_rate * np.dot( features_train_array[:, :].T , (labels_train_array - (np.exp(np.dot(features_train_array, old_weights)) / (1 + np.exp(np.dot(features_train_array, old_weights))))) )


        temp = np.exp(np.dot(features_test_array, weights))

        predictions = [0 if (1/(1+i)) > 0.5 else 1 for i in temp]

        predictions = np.array(predictions)

        MSE = (np.square(labels_test_array[:, 0] - predictions)).mean(axis=0)

        print("MSE for learn rate = " + str(learn_rate) + " : " + str(MSE))
        #print("the weights are: " + str(weights))



    # actual competition, it runs so fast that re runging it is no big deal.

    weights = np.zeros((4, 1))

    learn_rate = 0.001

    for _ in range(1000):
        old_weights = weights

        weights = old_weights + learn_rate * np.dot(features_train_array[:, :].T, (labels_train_array - (
                    np.exp(np.dot(features_train_array, old_weights)) / (
                        1 + np.exp(np.dot(features_train_array, old_weights))))))

    temp = np.exp(np.dot(features_test_array, weights))

    predictions = [0 if (1 / (1 + i)) > 0.5 else 1 for i in temp]

    predictions = np.array(predictions)


    TP, TN, FP, FN = confusion_matrix_maker(predictions, labels_test_array)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    negative_predictive_value = TN / (TN + FN)
    false_positive_rate = FP / (FP + TN)
    false_discovery_rate = FP / (FP + TP)

    F1 = 2 * (precision * recall) / (precision + recall)
    F2 = (1 + 4) * (precision * recall) / ((4 * precision) + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("\naccuracy for learn rate 0.001: " + str(accuracy)) # here to make sure there is not a mistake.

    print("\nprecision for learn rate 0.001: " + str(precision))
    print("recall for learn rate 0.001: " + str(recall))
    print("negative predictive value for learn rate 0.001: " + str(negative_predictive_value))
    print("false positive rate for learn rate 0.001: " + str(false_positive_rate))
    print("false discovery rate for learn rate 0.001: " + str(false_discovery_rate))

    print("\nF1 score for learn rate 0.001: " + str(F1))
    print("F2 score for learn rate 0.001: " + str(F2))




    print("\n***** Question 3.2: *****\n")


    weights = np.random.normal(0, 0.01, 4)

    learn_rate = 0.001

    for _ in range(1000):
        old_weights = weights

        batch = random.sample(range(0, len(labels_train_array)), 100)

        weights = old_weights + learn_rate * np.dot(features_train_array[batch, :].T, (labels_train_array[batch, 0] - (
                    np.exp(np.dot(features_train_array[batch, :], old_weights)) / (
                        1 + np.exp(np.dot(features_train_array[batch, :], old_weights))))))

    temp = np.exp(np.dot(features_test_array, weights))

    predictions = [0 if (1 / (1 + i)) > 0.5 else 1 for i in temp]

    predictions = np.array(predictions)


    TP, TN, FP, FN = confusion_matrix_maker(predictions, labels_test_array)
    accuracy = (TP + TN) / (TP + TN + FP + FN)



    micro_precision = (TP + TN) / (TP + FP + TN + FN)
    macro_precision = (TP / (TP + FP) + TN / (TN + FN)) / 2

    micro_recall = (TP + TN) / (TP + FN + TN + FP)
    macro_recall = (TP / (TP + FN) + TN / (TN + FP)) / 2

    micro_negative_predictive_value = (TN + TP) / (TN + FN + TP + FP)
    macro_negative_predictive_value = (TN / (TN + FN) + TP / (TP + FP)) / 2

    micro_false_positive_rate = (FP + FN) / (FP + TN + FN + TP)
    macro_false_positive_rate = (FP / (FP + TN) + FN / (FN + TP)) / 2

    micro_false_discovery_rate = (FP + FN) / (FP + TP + FN + TN)
    macro_false_discovery_rate = (FP / (FP + TP) + FN / (FN + TN)) / 2


    micro_F1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    macro_F1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    micro_F2 = (1 + 4) * (micro_precision * micro_recall) / ((4 * micro_precision) + micro_recall)
    macro_F2 = (1 + 4) * (macro_precision * macro_recall) / ((4 * macro_precision) + macro_recall)

    print("\naccuracy for learn rate 0.001: " + str(accuracy)) # here to make sure there is not a mistake.
    print("true positive count for learn rate 0.001: " + str(TP))
    print("true negative count for learn rate 0.001: " + str(TN))
    print("false positive count for learn rate 0.001: " + str(FP))
    print("false negative count for learn rate 0.001: " + str(FN))

    print("\nmicro precision for learn rate 0.001: " + str(micro_precision))
    print("macro precision for learn rate 0.001: " + str(macro_precision))

    print("\nmicro recall for learn rate 0.001: " + str(micro_recall))
    print("macro recall for learn rate 0.001: " + str(macro_recall))

    print("\nmicro negative predictive value for learn rate 0.001: " + str(micro_negative_predictive_value))
    print("macro negative predictive value for learn rate 0.001: " + str(macro_negative_predictive_value))

    print("\nmicro false positive rate for learn rate 0.001: " + str(micro_false_positive_rate))
    print("macro false positive rate for learn rate 0.001: " + str(macro_false_positive_rate))

    print("\nmicro false discovery rate for learn rate 0.001: " + str(micro_false_discovery_rate))
    print("macro false discovery rate for learn rate 0.001: " + str(macro_false_discovery_rate))

    print("\n\nmicro F1 score for learn rate 0.001: " + str(micro_F1))
    print("macro F1 score for learn rate 0.001: " + str(macro_F1))

    print("\nmicro F2 score for learn rate 0.001: " + str(micro_F2))
    print("macro F2 score for learn rate 0.001: " + str(macro_F2))


Question3()
