import os
import random

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from Resampler import Resampler
from copy import copy, deepcopy

# Used dataset macros
ONE_DOLLAR_G3_SYNTH = "dataset/GGG/1$/csv-1dollar-synth-best"
ONE_DOLLAR_G3_HUMAN = "dataset/GGG/1$/csv-1dollar-human"
N_DOLLAR_G3_SYNTH = "dataset/GGG/N$/csv-ndollar-synth-best"
N_DOLLAR_G3_HUMAN = "dataset/GGG/N$/csv-ndollar-human"


# DataLoader class, used for loading and saving the dataset and its attributes
# - 'test' mode:    Use only for test. It will load only the human gestures
# - 'train' mode:   Use only for training/validation. It will load only the synthetic gestures
# - 'full' mode:    Use when need to train and test at the same session. It will load everything
class DataLoader:

    # Constructor of the class
    def __init__(self, pad=True, resample=True,
                 normalize=True, include_fingerup=True, robust_normalization=True,
                 test_size=0.2, method='G3', dataset='1$', load_mode='full'):

        print('Starting the DataLoader construction ...')

        # Setting general data loader attributes
        self.use_padding = pad
        self.use_resampling = resample
        self.use_normalization = normalize
        self.include_fingerup = include_fingerup
        self.test_size = test_size
        self.method = method
        self.stroke_dataset = dataset
        self.load_mode = load_mode
        self.robust_normalization = robust_normalization

        print('.. Done with attribute settings. Loading the data ...')

        # Loading train, validation and test sets
        if self.load_mode is not 'test':
            self.train_set, self.validation_set, self.train_raw, self.validation_raw = self.__load_dataset_splitted(stroke_type='SYNTH')
            if self.load_mode == 'train':
                self.test_set = self.validation_set
                self.test_raw = self.validation_raw
        if self.load_mode is not 'train':
            self.test_set, self.test_raw = self.__load_dataset(stroke_type='HUMAN')
            if self.load_mode == 'test':
                self.train_set, self.validation_set = self.test_set, self.test_set
                self.train_raw, self.validation_raw = self.test_raw, self.test_raw

        self.raw_dataset = self.train_raw, self.validation_raw, self.test_raw
        self.raw_labels = self.train_set[1], self.validation_set[1], self.test_set[1]

        print('.. Done with data loading. Setting up classifier attributes ...')

        # Setting label repetition for the classifier
        if self.use_padding:
            self.labels_repeated = self.__get_labels_repeated()
            self.train_set_classifier = self.train_set[0], self.labels_repeated[0]
            self.validation_set_classifier = self.validation_set[0], self.labels_repeated[1]
            self.test_set_classifier = self.test_set[0], self.labels_repeated[2]

            print('.. Done with classifier attributes. Setting up regressor attributes ...')

            self.labels_regressed = self.__get_regressive_labels()
            self.train_set_regressor = self.train_set[0], self.labels_regressed[0]
            self.validation_set_regressor = self.validation_set[0], self.labels_regressed[1]
            self.test_set_regressor = self.test_set[0], self.labels_regressed[2]
        else:
            self.train_set_classifier, self.train_set_regressor = self.train_set, self.train_set
            self.validation_set_classifier, self.validation_set_regressor = self.validation_set, self.validation_set
            self.test_set_classifier, self.test_set_regressor = self.test_set, self.test_set

        self.tuple = self.train_set[0].shape[-1]

        print('Done with DataLoader construction!')

    # Function for reading a sample of the 1dollar class from csv
    def __read_csv_1dollar(self, file_name):
        df = pd.read_csv(file_name, sep=' ')
        df = df[['x', 'y']] / 100

        # Adding fingerup serie
        if self.include_fingerup:
            df['finger_up'] = 0
            df['finger_up'].iloc[-1] = 1

        return df.values.tolist()

    # Function for reading a sample of the 1dollar class from csv
    def __read_csv_ndollar(self, file_name):

        df = pd.read_csv(file_name, sep=' ')
        stroke_id = df['stroke_id']
        df = df[['x', 'y']] / 100
        df['stroke_id'] = stroke_id
        df = df[['stroke_id', 'x', 'y']]

        if self.include_fingerup:
            # Pre-processing finger up series
            stroke_ids = df['stroke_id'].to_numpy()
            stroke_ids_padded = np.concatenate((np.zeros(1, dtype='int64'), stroke_ids), axis=0)
            stroke_ids = np.concatenate((stroke_ids, np.zeros(1, dtype='int64')), axis=0)
            finger_up_padded = stroke_ids - stroke_ids_padded

            # Adding finger up series
            df['finger_up'] = finger_up_padded[1:]
            df['finger_up'].iloc[-1] = 1

        return df.values.tolist()

    # Function designed to load the $1 dataset
    def __load_dataset_on_folder(self, folder_name):

        tensor_x = []
        tensor_y = []
        labels_dict = {}
        index = 0

        files = os.listdir(folder_name)
        if self.load_mode == 'reduced':
            files = files[::30]

        i = 1
        for file in files:

            if i % 100 == 0:
                print("{}%- Loading on {}".format('{0:.2f}'.format(i/len(files)*100), folder_name))
            i += 1

            file_path = folder_name + '/' + file

            current_label = ''
            if self.method == 'G3':
                current_label = file.split('-')[1]

            if current_label not in labels_dict:
                labels_dict[current_label] = index
                index += 1

            if self.method == 'G3':
                if self.stroke_dataset == '1$':
                    tensor_x.append(self.__read_csv_1dollar(file_path))
                elif self.stroke_dataset == 'N$':
                    tensor_x.append(self.__read_csv_ndollar(file_path))

            tensor_y.append(labels_dict[current_label])

        print("Loading on {} completed.".format(folder_name))

        return tensor_x, tensor_y

    # Function designed to load the $1 dataset
    def __load_dataset(self, stroke_type, preprocess=True):

        # Computing dataset folder path
        dataset_folder = ''
        if self.stroke_dataset == '1$':
            if stroke_type == 'SYNTH':
                dataset_folder = ONE_DOLLAR_G3_SYNTH
            elif stroke_type == 'HUMAN':
                dataset_folder = ONE_DOLLAR_G3_HUMAN
        elif self.stroke_dataset == 'N$':
            if stroke_type == 'SYNTH':
                dataset_folder = N_DOLLAR_G3_SYNTH
            elif stroke_type == 'HUMAN':
                dataset_folder = N_DOLLAR_G3_HUMAN

        # Loading the dataset
        x, y = self.__load_dataset_on_folder(dataset_folder)
        x, y = np.array(x), np.array(y)

        x_raw = x

        if preprocess:
            x, y, x_raw = self.__preprocess_data(x, y, load_mode='test')

        return (x, y), x_raw

    # Function designed to load the $1 dataset splitted in test and train sets
    def __load_dataset_splitted(self, stroke_type):

        (x, y), x_raw = self.__load_dataset(stroke_type=stroke_type, preprocess=False)

        # Splitting into test and training set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=4)

        # Normalize x_train
        x_train, y_train, x_train_raw = self.__preprocess_data(x_train, y_train, augment_data=True, load_mode='train')

        # Normalize x_test
        x_test, y_test, x_test_raw = self.__preprocess_data(x_test, y_test, augment_data=False, load_mode='validation')

        return (x_train, y_train), (x_test, y_test), x_train_raw, x_test_raw

    def __preprocess_data(self, x, y, augment_data=False, load_mode=''):

        # Instantiating the resampler
        res = Resampler()

        if self.use_normalization:
            if augment_data:
                if self.robust_normalization:
                    x, y = self.__augment_data(x, y, res, load_mode=load_mode)
                else:
                    x = res.normalize_dataset(x, stroke_dataset=self.stroke_dataset,
                                              include_fingerup=self.include_fingerup)

            else:
                x = res.normalize_dataset(np.array(x), stroke_dataset=self.stroke_dataset,
                                          include_fingerup=self.include_fingerup, load_mode=load_mode)

        # Resampling branch
        if self.use_resampling:
            if self.stroke_dataset == 'N$':
                x = res.resample_ndollar_dataset(x, include_fingerup=self.include_fingerup)
            elif self.stroke_dataset == '1$':
                x = res.resample_onedollar_dataset(x, include_fingerup=self.include_fingerup)

        # Padding branch
        x_raw = x
        if self.use_padding:
            x = np.array(x)
            x = tf.keras.preprocessing.sequence.pad_sequences(x, padding="post", dtype='float32')

        return x, y, x_raw

    # Data augmentation function
    def __augment_data(self, x, y, res, load_mode=''):

        robustness = 5
        original_x = np.array(x)
        original_y = y

        for i in range(robustness - 1):

            curr_x = res.normalize_dataset(original_x, stroke_dataset=self.stroke_dataset,
                                           include_fingerup=self.include_fingerup, load_mode=load_mode)
            if i == 0:
                x = curr_x
            else:
                x = np.concatenate((x, curr_x), axis=0)
                y = np.concatenate((original_y, y), axis=0)

        return x, y

    def __get_labels_repeated(self):
        x_train, y_train = self.train_set
        x_test, y_test = self.test_set
        x_validation, y_validation = self.validation_set

        # Converting to numpy the label lists
        y_train = np.repeat(np.array(y_train), x_train.shape[1], axis=0).reshape((y_train.shape[0], x_train.shape[1]))
        y_validation = np.repeat(np.array(y_validation), x_validation.shape[1], axis=0).reshape((y_validation.shape[0],
                                                                                                 x_validation.shape[1]))
        y_test = np.repeat(np.array(y_test), x_test.shape[1], axis=0).reshape((y_test.shape[0], x_test.shape[1]))

        return y_train, y_validation, y_test

    def __get_regressive_labels(self):
        y_train, y_validation, y_test = self.__get_labels_repeated()

        y_train = self.__transform_regression_labels(np.array(y_train, dtype='float32'), self.train_raw)
        y_validation = self.__transform_regression_labels(np.array(y_validation, dtype='float32'), self.validation_raw)
        y_test = self.__transform_regression_labels(np.array(y_test, dtype='float32'), self.test_raw)

        # Only train and validation expand their dims because they need to be feed into the model
        y_train = np.expand_dims(y_train, axis=2)
        y_validation = np.expand_dims(y_validation, axis=2)

        return y_train, y_validation, y_test

    def __transform_regression_labels(self, ys, xs):

        for i in range(ys.shape[0]):

            # Computing padding
            padded_size, unpadded_size = ys[i].shape[0], xs[i].shape[0]
            pad_gap = padded_size - unpadded_size

            # Label transformation
            ys[i] = np.pad(np.arange(len(xs[i])), (0, pad_gap), 'constant', constant_values=(0, 0))
            ys[i] = np.array(ys[i] / len(xs[i]), dtype='float32')

        return ys

