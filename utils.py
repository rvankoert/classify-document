# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class utils:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def make_pairs(images, labels):
        # initialize two empty lists to hold the (image, image) pairs and
        # labels to indicate if a pair is positive or negative
        pairImages = []
        pairLabels = []
        # calculate the total number of classes present in the dataset
        # and then build a list of indexes for each class label that
        # provides the indexes for all examples with a given label
        uniqueLabels = np.unique(labels)
        numClasses = len(uniqueLabels)
        idx = [np.where(labels == uniqueLabels[i])[0] for i in range(0, numClasses)]
        # loop over all images
        for idxA in range(len(images)):
            # grab the current image and label belonging to the current
            # iteration
            currentImage = images[idxA]
            label = labels[idxA]
            # randomly pick an image that belongs to the *same* class
            # label
            labelIndex = np.where(uniqueLabels == label)[0][0]
            idxB = np.random.choice(idx[labelIndex])
            posImage = images[idxB]
            # prepare a positive pair and update the images and labels
            # lists, respectively
            # tf.keras.preprocessing.image.save_img('posImage1.png', tf.keras.preprocessing.image.array_to_img(currentImage))
            # tf.keras.preprocessing.image.save_img('posImage2.png', tf.keras.preprocessing.image.array_to_img(posImage))
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            negIdx = np.where(labels != label)[0]
            negImage = images[np.random.choice(negIdx)]
            # prepare a negative pair of images and update our lists
            # tf.keras.preprocessing.image.save_img('negImage1.png', tf.keras.preprocessing.image.array_to_img(currentImage))
            # tf.keras.preprocessing.image.save_img('negImage2.png', tf.keras.preprocessing.image.array_to_img(negImage))
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
        # return a 2-tuple of our image pairs and labels
        return (np.array(pairImages), np.array(pairLabels))

    def euclidean_distance(vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                           keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))

    def contrastive_loss(model1, model2, y, margin):
        with tf.name_scope("contrastive-loss"):
            distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keepdims=True))
            similarity = y * tf.square(distance)  # keep the similar label (1) close to each other
            dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance),
                                                           0))  # give penalty to dissimilar label if the distance is bigger than margin
            return tf.reduce_mean(dissimilarity + similarity) / 2

    def plot_training(H, plotPath):
        # construct a plot that plots and saves the training history
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H.history["loss"], label="train_loss")
        if "val_loss" in H.history:
            plt.plot(H.history["val_loss"], label="val_loss")
            plt.plot(H.history["val_accuracy"], label="val_acc")
        plt.plot(H.history["accuracy"], label="train_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plotPath)
