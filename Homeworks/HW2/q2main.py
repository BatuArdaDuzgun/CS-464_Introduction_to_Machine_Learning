import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Question2():

    labels = pd.read_csv("question-2-labels.csv")
    features = pd.read_csv("question-2-features.csv")

    y = labels_array = labels.to_numpy()
    features_array = features.to_numpy()
    feature_names = features.head()

    n, m = features_array.shape




    print("\n***** Question 2.2: *****\n")



    rank = np.linalg.matrix_rank(np.dot(features_array.T, features_array))


    print("X^T X is rank " + str(rank))





    print("\n***** Question 2.3: *****\n")


    X = np.ones((n, 2))
    X[:, 1] = features_array[:, 12]

    weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    print("w0: " + str(weights[0]))
    print("w1: " + str(weights[1]))


    predictions = np.dot(X, weights)


    MSE = (np.square(y - predictions)).mean(axis=0)


    print("\nthe MSE for linear regression with only LSTAT: " + str(MSE[0]))


    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], predictions, "bo", X[:, 1], y, "g^")
    plt.xlabel("LSTAT")
    plt.ylabel("the total price of the house")
    plt.title("LSTAT vs. house price labels and LSTAT vs. house price linear regression predictions")
    plt.legend(["predictions", "ground truth label"])
    plt.show()




    print("\n***** Question 2.4: *****\n")


    X = np.ones((n, 3))
    X[:, 1] = features_array[:, 12]
    X[:, 2] = np.square(features_array[:, 12])

    weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    print("w0: " + str(weights[0]))
    print("w1: " + str(weights[1]))
    print("w2: " + str(weights[2]))

    predictions = np.dot(X, weights)


    MSE = (np.square(y - predictions)).mean(axis=0)


    print("\nthe MSE for polynomial regression with only LSTAT: " + str(MSE[0]))


    plt.figure(figsize=(10, 6))
    plt.plot(X[:, 1], predictions, "bo", X[:, 1], y, "g^")
    plt.xlabel("LSTAT")
    plt.ylabel("the total price of the house")
    plt.title("LSTAT vs. house price labels and LSTAT vs. house price polynomial regression predictions")
    plt.legend(["predictions", "ground truth label"])
    plt.show()


Question2()
