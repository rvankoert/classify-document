import numpy as np
import keras
import tensorflow as tf
import config


class DataGenerator(keras.utils.Sequence):

    def preprocess(self, img, imgSize) -> np.ndarray:
        if img is None:
            img = np.zeros([imgSize[0], imgSize[1]], dtype=np.uint8)
            print("Image broken, zeroing")

        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_hue(img, 0.08)
        img = tf.image.random_saturation(img, 0.6, 1.6)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        # img = tf.keras.preprocessing.image.random_rotation(img,10)

        # img = tfio.experimental.image.decode_tiff(img)
        img = tf.image.rgb_to_grayscale(img)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # img = img / 255.0
        # img = 1 - img
        # img = tf.image.resize_with_pad(img, imgSize[0], imgSize[1])  # rescale to have matching height with target image
        img = tf.image.resize(img, (imgSize[0], imgSize[1]))  # rescale to have matching height with target image
        return img

    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(227, 227), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')

            X[i, ] = self.preprocess(ID, config.SHAPE)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
