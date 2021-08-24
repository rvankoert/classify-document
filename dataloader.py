import random
import numpy as np
import cv2
import tensorflow as tf
#import tensorflow_io as tfio

class Sample:
    "sample from the dataset"

    def __init__(self, filePath, label):
        self.label = label
        self.filePath = filePath


class DataLoaderRepublic:
    def preprocess(self, img, imgSize) -> np.ndarray:
        if img is None:
            img = np.zeros([imgSize[0], imgSize[1]], dtype=np.uint8)
            print("Image broken, zeroing")

        img = tf.io.read_file(img)
        img = tf.image.decode_png(img, 1)
        # img = tfio.experimental.image.decode_tiff(img)
        # img = tf.image.rgb_to_grayscale(img)
        # img = tf.image.convert_image_dtype(img, tf.float16)
        img = img / 255.0
        # img = 1 - img
        img = tf.image.resize_with_pad(img, imgSize[0], imgSize[1])  # rescale to have matching height with target image
        return img

    'Generates data for Keras'

    def __init__(self, filePath, imgSize):
        percentValidation = 0.15
        "loader for dataset at given location, preprocess images and labels according to parameters"
        self.dataAugmentation = False
        self.currIdx = 0
        self.imgSize = imgSize
        self.samples = []

        f = open(filePath)
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 2

            # filename
            fileName = lineSplit[0]
            label = lineSplit[1]

            # put sample into list
            self.samples.append(Sample(fileName, label))

        # split into training and validation set: 95% - 5%
        # random.seed(42)
        random.shuffle(self.samples)
        splitIdx = int((1.0 - percentValidation) * len(self.samples))
        #               self.trainSamples = self.samples[:splitIdx]
        tmpSamples = self.samples[:splitIdx]
        self.unknownTrainSamples = [i for i in tmpSamples[:splitIdx] if i.label == "unknown"]
        self.inferenceSamples = [i for i in tmpSamples[:splitIdx] if i.label != "unknown"]
        #               self.trainSamples = [print(i.label) for i in tmpSamples[:splitIdx]]
        self.validationSamples = self.samples[splitIdx:]
        self.validationSamples.extend(self.unknownTrainSamples)

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 2000

        # start with train set

    def split_data(self, images, labels, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

    def inferenceSet(self):
        "switch to randomly chosen subset of training set"
        self.currIdx = 0
        random.shuffle(self.inferenceSamples)
        self.samples = self.inferenceSamples[:self.numTrainSamplesPerEpoch]

        batchRange = range(0, len(self.samples))
        print(batchRange)
        #               print (self.samples[500])
        self.samples = [self.samples[i] for i in batchRange]

        batchRange = range(0, len(self.samples) - 1)
        print(batchRange)
        labels = [int(self.samples[i].label) for i in batchRange]
        filePaths = [self.samples[i].filePath for i in batchRange]
        imgs = [self.preprocess(self.samples[i].filePath, self.imgSize) for i in batchRange]
        # imgs = [self.preprocess('/media/rutger/906691c8-9441-4a4d-894e-c156384e5514/'+self.samples[i].filePath, self.imgSize) for i in batchRange]

        self.x_train, self.y_train, self.train_filePaths = np.array(imgs), np.array(labels), np.array(filePaths)
        # return (self.x_train, self.y_train)
        return (self.x_train, self.y_train, self.train_filePaths)

    def validationSet(self):
        "switch to validation set"
        return (self.x_valid, self.y_valid)


if __name__ == "__main__":
    print("test2")
