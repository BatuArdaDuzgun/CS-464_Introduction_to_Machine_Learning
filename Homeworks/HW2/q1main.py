import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Question1():

    images = pd.read_csv("images.csv")

    images_array = images.to_numpy()

    pixel_count, image_count = images_array.shape


    mean = sum(sum(images_array)) / (pixel_count * image_count)

    demean_images = images_array - mean

    demean_images = demean_images.T

    covariance_matrix = np.cov(demean_images)

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    explaind_variance = eigen_values / sum(eigen_values)



    pcs = np.zeros((10, 48, 48))

    print("\n***** Question 1.1: *****\n")

    for i in range(10):

        pcs[i, :, :] = np.reshape(eigen_vectors[:, i], (48, 48))
        plt.imshow(pcs[i, :, :], cmap="Greys")
        plt.title("PC " + str(i + 1) + ", PVE = " + str(explaind_variance[i]))
        print("PVE for PC " + str(i + 1) + " is " + str(explaind_variance[i]))
        plt.show()



    print("\n***** Question 1.2: *****\n")

    k = [1, 10, 50, 100, 500]

    for i in k:

        print("PVE for the first " + str(i) + " principal components: " + str(sum(explaind_variance[:i])))


    PVE = np.zeros(2304)
    k = range(2304)

    for i in k:

        PVE[i] = sum(explaind_variance[:i])

    plt.figure(figsize=(10, 6))
    plt.title("k vs. PVE")
    plt.xlabel("number of first principal components (log scale)")
    plt.ylabel("proportion of variance explained ")
    plt.xscale("log")
    plt.plot(k, PVE)
    plt.show()

    print("\n***** Question 1.3: *****\n")

    first_image = images_array[0, :]
    image_to_show = np.reshape(first_image, (48, 48))
    plt.imshow(image_to_show, cmap="Greys")
    plt.title("the original first image for comparison")
    plt.show()


    for k in [1, 10, 50, 100, 500]:

        PCAs = eigen_vectors[:, :k]

        low_dimesnsional_reprisentation = np.dot(first_image, PCAs)

        reconstruction = np.dot(PCAs, low_dimesnsional_reprisentation)

        image_to_show = np.reshape(reconstruction, (48, 48))
        plt.imshow(image_to_show, cmap="Greys")
        plt.title("Reconstructed first image using the\nfirst " + str(k) + " principal components")
        plt.show()


Question1()








































