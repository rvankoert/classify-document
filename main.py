import os
from collections import Counter, OrderedDict

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from augmenter import Augmenter
from utils import *
from model import *
import config
import sys

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import random
from pathlib import Path
from datagenerator import DataGenerator


def resize_with_aspect(input):
    return tf.image.resize_with_pad(
        input, config.SHAPE[0], config.SHAPE[1], method=ResizeMethod.BILINEAR,
        antialias=False
    )

def get_generator(set, training, class_indices):
    ids = []
    labels = []
    files = []

    for directory in os.scandir(set):
        if directory.is_dir():
            tmp_label = directory.name
            for file in os.scandir(os.path.join(set, tmp_label)):
                if file.is_file():
                    # print(file.path)
                    ids.append(file.path)
                    labels.append(tmp_label)
                    files.append((file.path, tmp_label))

    counter = 0
    add_classes = False
    if class_indices is None:
        tmp_class_indices = []
        add_classes = True
    else:
        tmp_class_indices = class_indices

    for tmp_label in labels:
        if tmp_label in tmp_class_indices:
            continue
        if add_classes:
            print('adding class: ' + tmp_label)
            tmp_class_indices.append(tmp_label)
        else:
            print("found new unknown label: " + tmp_label)
            exit()
        counter += 1
    tmp_classes = []
    for tmp_label in labels:
        tmp_classes.append(tmp_class_indices.index(tmp_label))

    num_classes = len(tmp_class_indices)

    new_train_files = []
    for file in files:
        id = file[0]
        tmp_label = tmp_class_indices.index(file[1])
        new_train_files.append((id, str(tmp_label)))
    train_files = new_train_files
    # print(train_files)
    # for file in train_files:
    #     file[1] =

    # train_generator = DataGenerator(ids, labels, height=args.height, width=args.width)

    train_batches = np.ceil(len(ids) / args.batch_size)
    generator = tf.data.Dataset.from_tensor_slices(train_files)
    if training:
        augmenter = Augmenter(
            do_shear=args.do_shear,
            do_speckle=args.do_speckle,
            do_rotate=args.do_rotate,
            do_elastic_transform=args.do_elastic_transform
        )
        data_augmentation = augmenter.create_augmentation_sequence()

        generator = (generator
                 .repeat()
                 .shuffle(len(train_files))
                 .map(lambda x: preprocess(x[0], x[1], args.height, args.width, args.channels, num_classes), num_parallel_calls=AUTOTUNE,
                    deterministic=False)
                 .map(lambda x, y: (data_augmentation(x, training=True), y))
                 .map(lambda x, y: (0.5 - (x / 255), y))
                 .batch(args.batch_size)
                 .prefetch(AUTOTUNE)
                 ).apply(tf.data.experimental.assert_cardinality(train_batches))
    else:
        generator = (generator
                 .map(lambda x: preprocess(x[0], x[1], args.height, args.width, args.channels, num_classes), num_parallel_calls=AUTOTUNE,
                    deterministic=False)
                 .map(lambda x, y: (0.5 - (x / 255), y))
                 .batch(args.batch_size)
                 .prefetch(AUTOTUNE)
                 ).apply(tf.data.experimental.assert_cardinality(train_batches))
    return generator, num_classes, tmp_class_indices, files, tmp_classes

@tf.function
def preprocess(img_path, label, height, width, channels, num_classes) -> np.ndarray:
    # img_path, label, height, width = data
    # print(img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels)
    print(type(img))
    # if augmenter is not None:
    # img = tf.image.random_brightness(img, 0.05)
    # img = tf.image.random_hue(img, 0.08)
    # img = tf.image.random_saturation(img, 0.6, 1.6)
    #     img = tf.image.random_contrast(img, 0.7, 1.3)
        # img = tf.keras.preprocessing.image.random_rotation(img, 0.2)

    # img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (height, width))  # rescale to have matching height with target image

    label = tf.strings.to_number(label, out_type='int32')
    categorical = tf.one_hot(label, depth=num_classes)

    return img, categorical



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used, use -1 for cpu')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.001,
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
parser.add_argument('--do_pretrain', help='do_pretrain', action='store_true')
parser.add_argument('--pre_train_epochs', metavar='pre_train_epochs', type=int, default=1,
                    help='pre_train_epochs to be used')
parser.add_argument('--do_train', help='train', action='store_true')
parser.add_argument('--use_class_weights', action='store_true',
                    help='use_class_weights')
parser.add_argument('--do_validation', action='store_true', help='validation')
parser.add_argument('--do_test', action='store_true', help='test')
parser.add_argument('--do_inference', action='store_true', help='inference')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default=None,
                    help='existing_model')
parser.add_argument('--loss', metavar='loss ', type=str, default="categorical_crossentropy",
                    help='categorical_crossentropy, binary_crossentropy, mse')
parser.add_argument('--optimizer', metavar='optimizer ', type=str, default='adamw',
                    help='optimizer: adam, adadelta, rmsprop, sgd, adamw. Default: adamw')
parser.add_argument('--train_set', metavar='train_set ', type=str, default='/home/rutger/data/republic/train/',
                    help='train_set to use for training')
parser.add_argument('--validation_set', metavar='validation_set ', type=str,
                    default=None,
                    help='validation_set to use for validation')
parser.add_argument('--test_set', metavar='test_set ', type=str, default='/home/rutger/data/republic/test/',
                    help='test_set to use for testing')
parser.add_argument('--inference_set', metavar='inference_set ', type=str,
                    default='/home/rutger/data/republic/inference/',
                    help='inference_set to use for inferencing')
parser.add_argument('--deterministic', action='store_true', help='deterministic')
parser.add_argument('--do_shear', action='store_true', help='augment training data with shear')
parser.add_argument('--do_rotate', action='store_true', help='augment training data with rotation')
parser.add_argument('--do_elastic_transform', action='store_true', help='augment training data with elastic transformation')
parser.add_argument('--do_speckle', action='store_true', help='augment training data with data distortion')


args = parser.parse_args()
options = vars(args)
print(options)

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.deterministic:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

validation_generator = None
classes = None
class_indices = None
classes_file = args.output + "/classes.txt"
if args.existing_model:
    with open(classes_file, 'r') as file:
        class_indices = eval(file.read().replace('\n', ''))

print(class_indices)
if args.validation_set:
    # validation_datagen = ImageDataGenerator(
    #     rescale=(1. / 255),
    #     # preserve_aspect_ratio=True
    #
    #     # preprocessing_function=tf.keras.applications.xception.preprocess_input
    # )
    #
    # validation_generatorold = validation_datagen.flow_from_directory(
    #     args.validation_set,
    #     target_size=(299,299),
    #     # preprocessing_function= resize_with_aspect,
    #     batch_size=args.batch_size,
    #     class_mode='categorical',
    #     classes=classes,
    #     shuffle=False
    # )  # set as validation data
    validation_generator, num_classes, class_indices, validation_files, validation_classes = get_generator(args.validation_set, False, class_indices)

    # print(class_indices)


if args.do_train:
    # train_datagen = ImageDataGenerator(
    #     rescale=(1. / 255),
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=20,
    #     zoom_range=[0.8, 1.2],
    #     brightness_range=[0.8, 1.2],
    #     channel_shift_range=20,
    #     # preprocessing_function=tf.keras.applications.xception.preprocess_input
    # )
    #
    # train_generator = train_datagen.flow_from_directory(
    #     args.train_set,
    #     target_size=(config.SHAPE[0], config.SHAPE[1]),
    #     # preprocessing_function= resize_with_aspect,
    #     batch_size=args.batch_size,
    #     color_mode=config.COLOR_MODE,
    #     class_mode='categorical',
    #     classes=classes,
    #     shuffle=True,
    # )  # set as training data

    # datalist = OrderedDict()
    training_generator, num_classes, class_indices_tmp, training_files, train_classes = get_generator(args.train_set, True, class_indices)

    if class_indices is None:
        class_indices = class_indices_tmp

    text_file = open(classes_file, "w")
    # print('train_generator.class_indices')
    # print(train_generator.class_indices)
    # print(train_generator2.class_indices)
    n = text_file.write(str(class_indices))
    text_file.close()

    monitor = 'accuracy'
    if args.do_validation:
        monitor = 'val_accuracy'
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='max')
    mcp_save = ModelCheckpoint(args.output + '/checkpoints/best_val/', save_best_only=True, monitor=monitor,
                               verbose=True, mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor=monitor, factor=0.6, patience=5, verbose=1, min_delta=1e-4,
                                       cooldown=3,
                                       mode='max')

    # counter = Counter(train_generator.classes)
    # print(train_generator.classes)
    # num_classes = len(counter)
    # max_val = float(max(counter.values()))
    # class_weights = None
    # if args.use_class_weights:
    #     class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
    class_weights = None
    # model = classifier.build_Xception_imagenet(config.SHAPE, num_classes)
    # model = classifier.build_vgg16_imagenet(config.SHAPE, num_classes)
    # model = classifier.build_classifier_model_E(config.SHAPE, num_classes)
    # model = classifier.build_classifier_model_F(config.SHAPE, num_classes)
    model = classifier.build_classifier_model_H((args.height, args.width, args.channels), num_classes)
    # model = classifier.build_classifier_model_I((args.height, args.width, args.channels), num_classes)

    for layer in model.layers:
        layer.trainable = False
    model.layers[-2].trainable = True
    model.layers[-1].trainable = True
    accuracy = ["accuracy"]

    loss = "mse"
    if args.loss == "binary_crossentropy":
        loss = "binary_crossentropy"
    elif args.loss == "categorical_crossentropy":
        loss = "categorical_crossentropy"
    elif args.loss == "sparse_categorical_crossentropy":
        loss = "sparse_categorical_crossentropy"
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
    if args.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, nesterov=False)
    if args.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    if args.optimizer == "adamw":
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=args.learning_rate)
    if args.optimizer == "adadelta":
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.learning_rate, rho=0.95, epsilon=1e-07,
                                              name="Adadelta")
    if args.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate)

    if args.existing_model:
        model_to_load = args.existing_model
        print('using model ' + model_to_load)
        model = tf.keras.models.load_model(model_to_load)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=accuracy)

    if args.do_pretrain:
        model.summary()
        # pretrain
        history = model.fit(training_generator,
                            validation_data=validation_generator,
                            epochs=args.pre_train_epochs,
                            callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                            class_weight=class_weights,
                            )

    for layer in model.layers:
        layer.trainable = True

    model.summary()
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        epochs=args.epochs,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        # class_weight=class_weights,
                        )

    model.save(args.output + '/models/last_model')
    # # plot the training history
    print("[INFO] plotting training history...")
    utils.plot_training(history, args.output + '/plot.png')

if args.do_validation:

    model = tf.keras.models.load_model(args.output + '/checkpoints/best_val')

    nb_samples = len(validation_files)
    label_map = class_indices

    # test_generator.reset()
    predict = model.predict(
        validation_generator
    )
    y_classes = predict.argmax(axis=-1)
    # for i in range(nb_samples):
    #     label = list(label_map.keys())[list(label_map.values()).index(y_classes[i])]
        # print("%s\t%s\t%s\t%s" % (filenames[i], label, y_classes[i], predict[i],))

    y_pred = np.argmax(predict, axis=1)
    # print(validation_classes)
    # print(len(validation_classes))
    # print(validation_generatorold.classes)
    # print(len(validation_generatorold.classes))
    # print(len(y_pred))
    print(class_indices)
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)
    print('Confusion Matrix')
    print(confusion_matrix(validation_classes, y_pred))
    print('Classification Report')
    target_names = class_indices
    print(classification_report(validation_classes, y_pred, target_names=target_names))

if args.do_test:
    model = tf.keras.models.load_model(args.output + '/checkpoints/best_val')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        args.test_set,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=args.batch_size,
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

if args.do_inference:
    print('inferencing...')
    model_to_load = args.output + '/checkpoints/best_val'
    if args.existing_model:
        model_to_load = args.existing_model

    print('using model ' + model_to_load)
    model = tf.keras.models.load_model(model_to_load)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    path = Path(args.inference_set)
    path = path.parent.absolute()
    target_class = os.path.basename(os.path.normpath(args.inference_set))
    test_generator = test_datagen.flow_from_directory(
        path,
        target_size=(config.SHAPE[0], config.SHAPE[1]),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=args.batch_size,
        classes=[target_class],
    )

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    with open(classes_file, 'r') as file:
        class_indices = eval(file.read().replace('\n', ''))
    print('using class mappings: ')
    print(class_indices)
    label_map = class_indices

    # test_generator.reset()
    predict = model.predict(
        test_generator
    )
    y_classes = predict.argmax(axis=-1)
    original_stdout = sys.stdout
    results_file = args.output + '/results.txt'
    print('writing results to ' + results_file)
    with open(results_file, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for i in range(nb_samples):
            try:
                label = list(label_map.keys())[list(label_map.values()).index(y_classes[i])]
                print("%s\t%s\t%s\t%s" % (filenames[i], label, y_classes[i], predict[i],))
            except:
                print("An exception occurred")

        sys.stdout = original_stdout
