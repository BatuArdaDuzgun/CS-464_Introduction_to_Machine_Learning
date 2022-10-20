import numpy as np
import pandas as pd
import time


def L1_distance(v1, v2):
    temp1 = v1 - v2
    temp2 = temp1[np.logical_not(np.isnan(temp1))]

    return sum(np.abs(temp2))


def L2_distance(v1, v2):
    temp1 = v1 - v2
    temp2 = temp1[np.logical_not(np.isnan(temp1))]

    return np.linalg.norm(temp2)


def KNN_for_one_point_L1(point, data, k):  # returns one or zero based on the data for one point

    temp_distances = np.zeros(len(data))

    for i in range(len(data)):
        distance = L1_distance(point[:-1], data[i, :-1])

        temp_distances[i] = distance

    order_indexes = np.argsort(temp_distances)

    prediction = 0

    for i in range(k):
        prediction += data[order_indexes[i], -1]

    if (prediction >= k / 2):

        return 1

    else:

        return 0



def KNN_for_one_point_L2(point, data, k):  # returns one or zero based on the data for one point

    temp_distances = np.zeros(len(data))

    for i in range(len(data)):
        distance = L2_distance(point[:-1], data[i, :-1])

        temp_distances[i] = distance

    order_indexes = np.argsort(temp_distances)

    prediction = 0

    for i in range(k):
        prediction += data[order_indexes[i], -1]

    if (prediction >= k / 2):

        return 1

    else:

        return 0



def KNN_for_set_L1(set, data, k):
    predictions = np.zeros(len(set))

    for i in range(len(set)):
        temp = KNN_for_one_point_L1(set[i, :], data, k)

        predictions[i] = temp

    return predictions


def KNN_for_set_L2(set, data, k):
    predictions = np.zeros(len(set))

    for i in range(len(set)):
        temp = KNN_for_one_point_L2(set[i, :], data, k)

        predictions[i] = temp

    return predictions


def KNN_results_L1(test, training, k):



    predic = KNN_for_set_L1(test[:,:], training[:,:], 9)

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



def KNN_results_L2(test, training, k):



    predic = KNN_for_set_L2(test[:,:], training[:,:], 9)

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



# =================================
# Training
# =================================

start = time.perf_counter()

diabetes_test_features = pd.read_csv("diabetes_test_features.csv")
diabetes_test_labels = pd.read_csv("diabetes_test_labels.csv")


diabetes_train_features = pd.read_csv("diabetes_train_features.csv")
diabetes_train_labels = pd.read_csv("diabetes_train_labels.csv")

diabetes_test_all = pd.merge(diabetes_test_features, diabetes_test_labels, on="Unnamed: 0")
diabetes_train_all = pd.merge(diabetes_train_features, diabetes_train_labels, on="Unnamed: 0")

diabetes_test_all.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
diabetes_train_all.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

cols_to_put_nan = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_test_all[cols_to_put_nan] = diabetes_test_all[cols_to_put_nan].replace({'0': np.nan, 0: np.nan})
diabetes_train_all[cols_to_put_nan] = diabetes_train_all[cols_to_put_nan].replace({'0': np.nan, 0: np.nan})

diabetes_test_all_np = diabetes_test_all.to_numpy()
diabetes_train_all_np = diabetes_train_all.to_numpy()

diabetes_test_all_means = np.nanmean(diabetes_test_all_np, axis=0)
diabetes_train_all_means = np.nanmean(diabetes_train_all_np, axis=0)

diabetes_test_all_means[0] = 0  # so that we don't de-mean the indexes or the results
diabetes_test_all_means[-1] = 0

diabetes_train_all_means[0] = 0
diabetes_train_all_means[-1] = 0

diabetes_test_all_std = np.nanstd(diabetes_test_all_np, axis=0)
diabetes_train_all_std = np.nanstd(diabetes_train_all_np, axis=0)

diabetes_test_all_std[0] = 1  # so that we don't standardize the indexes or the results
diabetes_test_all_std[-1] = 1

diabetes_train_all_std[0] = 1
diabetes_train_all_std[-1] = 1

# standerzing the data:
diabetes_test_all_np = diabetes_test_all_np - diabetes_test_all_means
diabetes_test_all_np = diabetes_test_all_np / diabetes_test_all_std

diabetes_train_all_np = diabetes_train_all_np - diabetes_train_all_means
diabetes_train_all_np = diabetes_train_all_np / diabetes_train_all_std

#the indexes are never used so we will remove them.
diabetes_test_no_index_np = diabetes_test_all_np[:, 1:]
diabetes_train_no_index_np = diabetes_train_all_np[:, 1:]

finish = time.perf_counter()
print("The training (opening the files and normalizing them) computation time: " + str(finish - start) + "\n")


# =================================
# Testing L1
# =================================

start = time.perf_counter()
sucses, tp, tn, fp, fn = KNN_results_L1(diabetes_test_no_index_np, diabetes_train_no_index_np, 9)
finish = time.perf_counter()
print("original L1 norm succes: " + str(sucses))
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("with computation time: " + str(finish - start) + "\n")

removed_colum_indexs = []
new_success = 0

best = sucses

for _ in range(8):

    indexToRemove = -1
    best = 0

    start = time.perf_counter()

    for I in range(0, (diabetes_test_no_index_np.shape[1] - 1)):

        if (I not in removed_colum_indexs):

            temp_removed_colum_indexs = removed_colum_indexs + [I]

            temp_test = np.delete(diabetes_test_no_index_np, temp_removed_colum_indexs, 1)
            temp_train = np.delete(diabetes_train_no_index_np, temp_removed_colum_indexs, 1)

            temp_sucses, temp_tp, temp_tn, temp_fp, temp_fn = KNN_results_L1(temp_test, temp_train, 9)

            if (best < temp_sucses):

                best = temp_sucses
                tp, tn, fp, fn = temp_tp, temp_tn, temp_fp, temp_fn
                indexToRemove = I

    finish = time.perf_counter()

    elapsed_time = (finish - start) / (diabetes_test_no_index_np.shape[1] - 1 - len(removed_colum_indexs))

    removed_colum_indexs.append(indexToRemove)

    removed_futures = ""
    for i in removed_colum_indexs:

        removed_futures += (str(diabetes_train_all.columns[i + 1]) + ", ")

    print("L1 norm success without " + removed_futures + ": " + str(best))
    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("with test computation time: " + str(elapsed_time) + "\n")


# =================================
# Testing L2
# =================================

start = time.perf_counter()
sucses, tp, tn, fp, fn = KNN_results_L2(diabetes_test_no_index_np, diabetes_train_no_index_np, 9)
finish = time.perf_counter()
print("original l2 norm success: " + str(sucses))
print("TP: " + str(tp))
print("TN: " + str(tn))
print("FP: " + str(fp))
print("FN: " + str(fn))
print("with computation time: " + str(finish - start) + "\n")

removed_colum_indexs = []
new_success = 0

best = sucses

for _ in range(8):

    indexToRemove = -1
    best = 0

    start = time.perf_counter()

    for I in range(0, (diabetes_test_no_index_np.shape[1] - 1)):

        if (I not in removed_colum_indexs):

            temp_removed_colum_indexs = removed_colum_indexs + [I]

            temp_test = np.delete(diabetes_test_no_index_np, temp_removed_colum_indexs, 1)
            temp_train = np.delete(diabetes_train_no_index_np, temp_removed_colum_indexs, 1)

            temp_sucses, temp_tp, temp_tn, temp_fp, temp_fn = KNN_results_L2(temp_test, temp_train, 9)

            if (best < temp_sucses):

                best = temp_sucses
                tp, tn, fp, fn = temp_tp, temp_tn, temp_fp, temp_fn
                indexToRemove = I

    finish = time.perf_counter()

    elapsed_time = (finish - start) / (diabetes_test_no_index_np.shape[1] - 1 - len(removed_colum_indexs))

    removed_colum_indexs.append(indexToRemove)

    removed_futures = ""
    for i in removed_colum_indexs:

        removed_futures += (str(diabetes_train_all.columns[i + 1]) + ", ")

    print("L2 norm success without " + removed_futures + ": " + str(best))
    print("TP: " + str(tp))
    print("TN: " + str(tn))
    print("FP: " + str(fp))
    print("FN: " + str(fn))
    print("with test computation time: " + str(elapsed_time) + "\n")


