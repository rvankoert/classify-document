from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception


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
            tf.keras.layers.Dropout(0.2),
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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation=tf.keras.layers.LeakyReLU(alpha=0.3)),
            tf.keras.layers.Dense(numClasses)
        ])

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
        model = tf.keras.Sequential()
        # model.add(VGG16(weights='imagenet'))
        # VGG = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)
        # VGG = VGG16(weights='imagenet')

        model = Xception(include_top=False, weights='imagenet', input_shape=inputShape)

        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # for layer in model.layers[:10]:
        #     layer.trainable = False
        # flat1 = Flatten()()
        flat1 = GlobalAveragePooling2D(name="GlobalAveragePooling1")(model.layers[-1].output)

        # dropout1 = Dropout(0.5)(flat1)
        flat2 = Dense(1024, name="fc_dense1", activation="elu")(flat1)
        dropout2 = Dropout(0.5)(flat2)
        class1 = Dense(1024, name="fc_dense2", activation="elu")(dropout2)
        dropout3 = Dropout(0.5)(class1)
        output = Dense(numClasses, activation='softmax')(dropout3)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)
        for layer in model.layers[:126]:
            if layer.name == 'fc_dense1':
                break
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
