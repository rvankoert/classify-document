from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Input
from tensorflow.keras.layers import Add,Concatenate, Conv2D, GlobalMaxPooling2D, MaxPooling2D, ReLU, BatchNormalization, AveragePooling2D


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

class classifier:
    def build_classifier_model_A(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(96, (11, 11), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(numClasses)
        ])

        return model

    def build_classifier_model_B(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(numClasses)
        ])

        return model

    def build_classifier_model_C(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(96, (11, 11), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dense(numClasses)
        ])

        return model

    # https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICPR-2014/data/5209d168.pdf
    # le Kang
    # requires 150x150 input
    def build_classifier_model_D(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(20, (7, 7), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(50, (5, 5), strides=(1, 1),
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dense(numClasses)
        ])

        return model

    # alexnet based
    # https://arxiv.org/pdf/1711.05862.pdf
    # input off 227x227
    def build_classifier_model_E(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dense(numClasses)
        ])

        return model

    def build_classifier_model_F(inputShape, numClasses):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation='elu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same",
                                   activation='elu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same",
                                   activation='elu'),
            tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding="same",
                                   activation='elu'),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                                   activation='elu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(192, (3, 3), strides=(1, 1), padding="same",
                                   activation='elu'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='elu'),
            tf.keras.layers.Dense(4096, activation='elu'),
            tf.keras.layers.Dense(numClasses, activation='softmax')
        ])

        return model

    # resnet-like
    def build_classifier_model_G(inputShape, numClasses):
        initializer = tf.keras.initializers.GlorotNormal(42)

        input_img = Input(
            shape=inputShape, name="image"
        )
        x1 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(input_img)

        x2 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 =Add()([x1,x3])
        x1 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(x4)

        x2 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=64, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 = Add()([x4,x3])
        x4 = MaxPooling2D(pool_size=(4, 4), strides=(3, 3))(x4)
        x1 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(x4)

        x2 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 = Add()([x1,x3])
        x1 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(x4)

        x2 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 = Add()([x4,x3])
        x4 = MaxPooling2D(pool_size=(4, 4), strides=(3, 3))(x4)
        x1 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(x4)

        x2 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 = Add()([x1,x3])
        x1 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv1",
                kernel_initializer=initializer
            )(x4)

        x2 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv2",
                kernel_initializer=initializer
            )(x1)

        x3 = Conv2D(filters=256, kernel_size=[3, 3], strides=(1, 1),
                activation='elu',
                padding='same',
                # name="Conv3",
                kernel_initializer=initializer
            )(x2)
        x4 = Add()([x4,x3])
        x4 = MaxPooling2D(pool_size=(4, 4), strides=(3, 3))(x4)
        x1 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv1",
                    kernel_initializer=initializer
                    )(x4)

        x2 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv2",
                    kernel_initializer=initializer
                    )(x1)

        x3 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv3",
                    kernel_initializer=initializer
                    )(x2)
        x4 = Add()([x1, x3])
        x1 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv1",
                    kernel_initializer=initializer
                    )(x4)

        x2 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv2",
                    kernel_initializer=initializer
                    )(x1)

        x3 = Conv2D(filters=512, kernel_size=[3, 3], strides=(1, 1),
                    activation='elu',
                    padding='same',
                    # name="Conv3",
                    kernel_initializer=initializer
                    )(x2)
        x4 = Add()([x4, x3])
        x4 = MaxPooling2D(pool_size=(4, 4), strides=(3, 3))(x4)
        # x4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x4)

        flat1 = GlobalAveragePooling2D()(x4)
        flat2 = GlobalMaxPooling2D()(x4)
        flat1 = Concatenate()([flat1,flat2])
        # flat1 = Flatten()(x4)
        flat2 = Dense(1024, activation='elu')(flat1)
        # dropout2 = Dropout(0.5)(flat2)
        class1 = Dense(1024, activation='elu')(flat2)
        # dropout3 = Dropout(0.5)(class1)
        output = Dense(numClasses, activation='softmax')(class1)
        # define new model
        model = Model(inputs=input_img, outputs=output)

        return model



    def build_classifier_model_H(inputShape, num_classes):
        initializer = tf.keras.initializers.GlorotNormal(42)

        inputs = Input(
            shape=inputShape, name="image"
        )
        num_filters = 16

        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=3,
                   strides=1,
                   filters=num_filters,
                   padding="same")(t)
        t = relu_bn(t)

        num_blocks_list = [2, 5, 5, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
            num_filters *= 2

        t = AveragePooling2D(4)(t)
        t = Flatten()(t)
        t = Dense(1024, activation='elu')(t)
        t = Dense(1024, activation='elu')(t)

        outputs = Dense(num_classes, activation='softmax')(t)

        model = Model(inputs, outputs)

        return model

    def build_classifier_model_I(inputShape, num_classes):
        initializer = tf.keras.initializers.GlorotNormal(42)

        inputs = Input(
            shape=inputShape, name="image"
        )
        num_filters = 16

        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=3,
                   strides=1,
                   filters=num_filters,
                   padding="same")(t)
        t = relu_bn(t)

        num_blocks_list = [2, 5, 5, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
            num_filters *= 2

        t = GlobalMaxPooling2D()(t)
        t = Flatten()(t)
        t = Dense(1024, activation='elu')(t)
        t = Dense(1024, activation='elu')(t)

        outputs = Dense(num_classes, activation='softmax')(t)

        model = Model(inputs, outputs)

        return model

    # reguires rgb
    def build_vgg16_imagenet(inputShape, numClasses):
        model = tf.keras.Sequential()
        # model.add(VGG16(weights='imagenet'))
        # VGG = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        # VGG = VGG16(weights='imagenet')

        model = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)

        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # flat1 = Flatten()()
        flat1 = GlobalAveragePooling2D()(model.layers[-1].output)

        # dropout1 = Dropout(0.5)(flat1)
        flat2 = Dense(1024, activation="elu")(flat1)
        dropout2 = Dropout(0.5)(flat2)
        class1 = Dense(1024, activation="elu")(dropout2)
        dropout3 = Dropout(0.5)(class1)
        output = Dense(numClasses, activation='softmax')(dropout3)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # for layer in VGG.layers[:10]:
        #     layer.trainable = False
        # for layer in VGG.layers[:len(VGG.layers)-3]:
        #     model.add(layer)
        # model.add(Flatten())
        # model.add(VGG)
        # model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # # model.add(Dropout(0.5))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(numClasses))
        return model

        # reguires rgb

    def build_vgg16_imagenet_dropout(inputShape, numClasses):
        model = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)

        x = model.layers[-19]
        x = model.layers[-18](x.output)
        x = Dropout(0.2)(x)
        x = model.layers[-17](x)
        x = Dropout(0.2)(x)
        x = model.layers[-16](x)  # MP
        x = model.layers[-15](x)
        x = Dropout(0.2)(x)
        x = model.layers[-14](x)
        x = Dropout(0.2)(x)
        x = model.layers[-13](x)  # MP
        x = model.layers[-12](x)
        x = Dropout(0.2)(x)
        x = model.layers[-11](x)
        x = Dropout(0.2)(x)
        x = model.layers[-10](x)
        x = Dropout(0.2)(x)
        x = model.layers[-9](x)  # MP
        x = model.layers[-8](x)
        x = Dropout(0.2)(x)
        x = model.layers[-7](x)
        x = Dropout(0.2)(x)
        x = model.layers[-6](x)
        x = Dropout(0.2)(x)
        x = model.layers[-5](x)  # MP
        x = model.layers[-4](x)
        x = Dropout(0.2)(x)
        x = model.layers[-3](x)
        x = Dropout(0.2)(x)
        x = model.layers[-2](x)
        x = Dropout(0.2)(x)
        x = model.layers[-1](x)

        flat1 = Flatten()(x)

        # dropout1 = Dropout(0.5)(flat1)
        flat2 = Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(flat1)
        dropout2 = Dropout(0.5)(flat2)
        class1 = Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(dropout2)
        dropout3 = Dropout(0.5)(class1)
        output = Dense(numClasses, activation='softmax')(dropout3)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        print(len(model.layers))
        for layer in model.layers[:31]:
            layer.trainable = False

        # for layer in VGG.layers[:10]:
        #     layer.trainable = False
        # for layer in VGG.layers[:len(VGG.layers)-3]:
        #     model.add(layer)
        # model.add(Flatten())
        # model.add(VGG)
        # model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # # model.add(Dropout(0.5))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(numClasses))
        return model

    def build_Xception_imagenet(inputShape, numClasses):
        # model = tf.keras.Sequential()
        # model.add(VGG16(weights='imagenet'))
        # VGG = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        # VGG = VGG16(weights='imagenet')

        model = Xception(include_top=False, weights='imagenet', input_shape=inputShape, classes=numClasses)
        # model = Xception(include_top=False, weights=None, input_shape=inputShape, classes=numClasses, pooling=None)

        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # flat1 = Flatten()()
        # flat1 = Flatten()(model.layers[-1].output)
        # flat2 = Dense(1024, name="fc_dense1", activation="elu")(flat1)
        # output = Dense(numClasses, activation='softmax')(flat2)

        model.summary()
        # flat1 = GlobalAveragePooling2D(name="GlobalAveragePooling1")(model.layers[-1].output)

        flat1 = Flatten()(model.layers[-1].output)
        # # dropout1 = Dropout(0.5)(flat1)
        flat2 = Dense(1024, name="fc_dense1", activation="elu")(flat1)
        # # flat2 = Dropout(0.5)(flat2)
        # class1 = Dense(1024, name="fc_dense2", activation="elu")(flat2)
        # # class1 = Dropout(0.5)(class1)
        output = Dense(numClasses, activation='softmax')(flat2)
        # # define new model
        model = Model(inputs=model.inputs, outputs=output)
        # for layer in model.layers[:126]:
        #     if layer.name == 'fc_dense1':
        #         break
        #     layer.trainable = False

        # for layer in VGG.layers[:10]:
        #     layer.trainable = False
        # for layer in VGG.layers[:len(VGG.layers)-3]:
        #     model.add(layer)
        # model.add(Flatten())
        # model.add(VGG)
        # model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # # model.add(Dropout(0.5))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(numClasses))
        return model

    def build_classifier(inputShape, numClasses):
        # specify the inputs for the feature extractor network
        # inputs = Sequential(inputShape)
        # x = Conv2D(8, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(inputs)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # x = Conv2D(16, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # # x = Dropout(0.3)(x)
        # x = Conv2D(32, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # # x = Conv2D(256, (3, 3), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        # # x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
        # # x = Dropout(0.3)(x)
        #
        # x = Flatten()(x)
        #
        # x = Dense(128)(x)
        # x = Dense(128)(x)
        # x = Dense(numClasses)(x)
        # outputs = x
        # model = Model(inputs, outputs)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=inputShape),

            tf.keras.layers.Conv2D(96, (11, 11), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same",
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(numClasses)
        ])

        return model
