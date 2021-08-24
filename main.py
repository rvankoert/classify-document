import os
from collections import Counter

from keras_preprocessing.image import ImageDataGenerator
import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from utils import *
from model import *
import config
import sys

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import random


def resize_with_aspect(input):
    return tf.image.resize_with_pad(
        input, config.SHAPE[0], config.SHAPE[1], method=ResizeMethod.BILINEAR,
        antialias=False
    )


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used, use -1 for cpu')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.0001,
                    help='learning_rate to be used')
parser.add_argument('--epochs', metavar='epochs', type=int, default=5,
                    help='epochs to be used')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=16,
                    help='batch_size to be used, when using variable sized input this must be 1')

parser.add_argument('--height', metavar='height', type=int, default=299,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=299,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=3,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--pretrain', metavar='pretrain', type=bool, default=False,
                    help='pretrain')
parser.add_argument('--pre_train_epochs', metavar='pre_train_epochs', type=int, default=5,
                    help='pre_train_epochs to be used')
parser.add_argument('--train', metavar='train', type=bool, default=False,
                    help='train')
parser.add_argument('--use_class_weights', metavar='use_class_weights', type=bool, default=False,
                    help='use_class_weights')
parser.add_argument('--validate', metavar='validate', type=bool, default=False,
                    help='validate')
parser.add_argument('--test', metavar='test', type=bool, default=False,
                    help='test')
parser.add_argument('--inference', metavar='inference', type=bool, default=False,
                    help='inference')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                    help='existing_model')
parser.add_argument('--loss', metavar='loss ', type=str, default="binary_crossentropy",
                    help='binary_crossentropy, mse')
parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='rmsprop',
                    help='optimizer: adam, adadelta, rmsprop, sgd')
parser.add_argument('--memory_limit', metavar='memory_limit ', type=int, default=4096,
                    help='memory_limit for gpu. Default 4096')
parser.add_argument('--train_set', metavar='train_set ', type=str, default='/home/rutger/data/republic/train/',
                    help='train_set to use for training')
parser.add_argument('--validation_set', metavar='validation_set ', type=str,
                    default='/home/rutger/data/republic/validation/',
                    help='validation_set to use for validation')
parser.add_argument('--test_set', metavar='test_set ', type=str, default='/home/rutger/data/republic/test/',
                    help='test_set to use for testing')
parser.add_argument('--inference_set', metavar='inference_set ', type=str,
                    default='/home/rutger/data/republic/inference/',
                    help='inference_set to use for inferencing')

args = parser.parse_args()
options = vars(args)
print(options)
# config.SEED = args.seed
# config.GPU = args.gpu
# config.LEARNING_RATE = args.learning_rate
# config.IMG_SHAPE = (args.height, args.width, args.channels)
# config.BATCH_SIZE = args.batch_size
# config.EPOCHS = args.epochs
# config.BASE_OUTPUT = args.output
# config.MODEL_NAME = args.model_name
# config.EXISTING_MODEL = args.existing_model
# config.OPTIMIZER = args.optimizer
# config.LOSS = args.loss

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
if args.gpu >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if (len(gpus) > 0):
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

validation_generator = None
classes = None
if args.validate:
    validation_datagen = ImageDataGenerator(
        rescale=(1. / 255)
    )

    validation_generator = validation_datagen.flow_from_directory(
        args.validation_set,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        # preprocessing_function= resize_with_aspect,
        batch_size=args.batch_size,
        color_mode=config.COLOR_MODE,
        class_mode='categorical',
        classes=classes,
        shuffle=False
    )  # set as validation data

classes_file = args.output+"/classes.txt"
if args.train:
    train_datagen = ImageDataGenerator(
        rescale=(1. / 255),
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,
        zoom_range=[0.9, 1.1],
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_set,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        # preprocessing_function= resize_with_aspect,
        batch_size=args.batch_size,
        color_mode=config.COLOR_MODE,
        class_mode='categorical',
        classes=classes,
        shuffle=True,
    )  # set as training data

    # print(train_generator.class_indices.keys())
    text_file = open(classes_file, "w")
    n = text_file.write(str(train_generator.class_indices))
    text_file.close()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='max')
    mcp_save = ModelCheckpoint(args.output+'/checkpoints/best_val/', save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=5, verbose=1, min_delta=1e-4,
                                       cooldown=3,
                                       mode='max')

    counter = Counter(train_generator.classes)
    num_classes = len(counter)
    max_val = float(max(counter.values()))
    class_weights = None
    if args.use_class_weights:
        class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

    model = classifier.build_Xception_imagenet(config.SHAPE, num_classes)

    accuracy = "accuracy"

    loss = "mse"
    if args.loss == "binary_crossentropy":
        loss = "binary_crossentropy"

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
    if args.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
    if args.optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    if args.optimizer == "adadelta":
        optimizer = keras.optimizers.Adadelta(learning_rate=args.learning_rate, rho=0.95, epsilon=1e-07,
                                              name="Adadelta")
    if args.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=accuracy)

    if args.pretrain:
        model.summary()
        # pretrain
        history = model.fit(train_generator,
                            validation_data=validation_generator,
                            epochs=args.pre_train_epochs,
                            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                            class_weight=class_weights,
                            # use_multiprocessing=True,
                            # workers=16
                            )

    for layer in model.layers:
        layer.trainable = True

    model.summary()
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=args.epochs,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        class_weight=class_weights,
                        # use_multiprocessing=True,
                        # workers=16
                        )

    model.save(args.output+'/models/last_model')
    # # plot the training history
    print("[INFO] plotting training history...")
    utils.plot_training(history, args.output+'/plot.png')

if args.validate:

    model = keras.models.load_model(args.output+'/checkpoints/best_val')

    with open(classes_file, 'r') as file:
        class_indices = eval(file.read().replace('\n', ''))

    filenames = validation_generator.filenames
    nb_samples = len(filenames)
    label_map = class_indices

    # test_generator.reset()
    predict = model.predict(
        validation_generator
    )
    y_classes = predict.argmax(axis=-1)
    for i in range(nb_samples):
        label = list(label_map.keys())[list(label_map.values()).index(y_classes[i])]
        print("%s\t%s\t%s\t%s" % (filenames[i], label, y_classes[i], predict[i],))

    y_pred = np.argmax(predict, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = class_indices.keys()
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

if args.test:
    model = keras.models.load_model('checkpoints/best_val')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        # '/home/rutger/republic/randomprinttest/',
        args.test,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=args.batch_size,
        classes=["statengeneraalpilotall"],
    )

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    # label_map = (train_generator.class_indices)
    with open(classes_file, 'r') as file:
        class_indices = eval(file.read().replace('\n', ''))
    label_map = class_indices
    # test_generator.reset()
    predict = model.predict(
        test_generator
    )
    y_classes = predict.argmax(axis=-1)
    original_stdout = sys.stdout
    with open('results.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for i in range(nb_samples):
            try:
                label = list(label_map.keys())[list(label_map.values()).index(y_classes[i])]
                print("%s\t%s\t%s\t%s" % (filenames[i], label, y_classes[i], predict[i],))
            except:
                print("An exception occurred")

        sys.stdout = original_stdout

if args.inference:
    model = keras.models.load_model(args.output+'/checkpoints/best_val')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        # '/home/rutger/republic/randomprinttest/',
        args.inference_set,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=args.batch_size,
        classes=["data"],
    )

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    with open(classes_file, 'r') as file:
        class_indices = eval(file.read().replace('\n', ''))

    label_map = class_indices

    # test_generator.reset()
    predict = model.predict(
        test_generator
    )
    y_classes = predict.argmax(axis=-1)
    original_stdout = sys.stdout
    with open('results.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for i in range(nb_samples):
            try:
                label = list(label_map.keys())[list(label_map.values()).index(y_classes[i])]
                print("%s\t%s\t%s\t%s" % (filenames[i], label, y_classes[i], predict[i],))
            except:
                print("An exception occurred")

        sys.stdout = original_stdout
