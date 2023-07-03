import tensorflow as tf
import elasticdeform.tf as etf
import tensorflow_addons as tfa


class Augmenter(tf.keras.utils.Sequence):

    def __init__(self, seed: int = 1, do_elastic_transform: bool = False, do_shear: bool = False,
                 do_speckle: bool = False, do_rotate: bool = False):
        self.seed = seed
        self.do_rotate = do_rotate
        self.do_speckle = do_speckle
        self.do_shear = do_shear
        self.do_elastic_transform = do_elastic_transform
        self.height = 64
        self.channels = 1
        tf.random.set_seed(seed)

    @tf.function
    def elastic_transform(self, original: tf.Tensor) -> tf.Tensor:
        displacement_val = tf.random.normal([2, 3, 3]) * 5

        deformed = etf.deform_grid(original, displacement_val, axis=(0, 1), order=3)

        return deformed

    @tf.function
    def shear(self, image: tf.Tensor) -> tf.Tensor:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        channels = image.shape[2]
        image = tf.image.resize_with_pad(image, image_height, image_width + 64 + 50)
        random_shear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

        if channels == 4:
            # crappy workaround for bug in shear_x where alpha causes errors
            channel1, channel2, channel3, alpha = tf.split(image, 4, axis=2)
            image = tf.concat([channel1, channel2, channel3], axis=2)
            image = tfa.image.shear_x(image, random_shear, replace=0)
            image2 = tf.concat([alpha, alpha, alpha], axis=2)
            image2 = tfa.image.shear_x(image2, random_shear, replace=0)
            channel1, channel2, channel3 = tf.split(image, 3, axis=2)
            alpha, alpha, alpha = tf.split(image2, 3, axis=2)
            image = tf.concat([channel1, channel2, channel3, alpha], axis=2)
        elif channels == 3:
            image = tfa.image.shear_x(image, random_shear, replace=0)
        else:
            # channel1 = tf.split(image, 1, axis=-1)
            image = tf.concat([image, image, image], axis=2)
            image = tfa.image.shear_x(image, random_shear, replace=0)
            image, image, image = tf.split(image, 3, axis=2)

        return tf.image.crop_to_bounding_box(image, int((tf.shape(image)[0] - image_height) / 2),
                                             int((tf.shape(image)[1] - image_width) / 2), image_height, image_width)

    def create_augmentation_sequence(self) -> tf.keras.Sequential:
        sequential = tf.keras.Sequential()
        sequential.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)))

        # shear has to be before elastic transform, else it does not work
        if self.do_shear:
            shear_lambda = tf.keras.layers.Lambda(lambda x: self.shear(x))
            sequential.add(shear_lambda)

        if self.do_rotate:
            rotation = tf.keras.layers.RandomRotation(0.05, seed=self.seed, fill_mode="constant")
            sequential.add(rotation)

        if self.do_elastic_transform:
            et_lambda = tf.keras.layers.Lambda(lambda x: self.elastic_transform(x))
            sequential.add(et_lambda)

        if self.do_speckle:
            dropout = tf.keras.layers.Dropout(.05, seed=self.seed)
            sequential.add(dropout)

        sequential.add(tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.uint8)))

        return sequential





