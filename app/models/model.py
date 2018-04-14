from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
import numpy as np


class Model:
    def __init__(self,
                 optimizer,
                 learning_rate,
                 decay,
                 momentum,
                 kernel_regularization_params,
                 activity_regularization_params,
                 epochs,
                 batch_size):

        self.model = None

        self.epochs = epochs
        self.batch_size = batch_size

        self._x_train = None
        self._partial_x_train = None
        self._partial_y_train = None
        self._x_val = None
        self._y_val = None

        # restrictions about the optimizers and the loss functions.
        self.optimizer = optimizer
        if self.optimizer == 'sgd':
            opt = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

        elif self.optimizer == 'rmsprop':
            opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=decay)

        elif self.optimizer == 'adagrad':
            # defaults: lr=0.01, epsilon=1e-08, decay=0.0
            opt = optimizers.Adagrad(lr=learning_rate, epsilon=1e-08, decay=decay)

        elif self.optimizer == 'adadelta':
            # defaults: lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0
            opt = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=1e-08, decay=decay)

        elif self.optimizer == 'adam':
            # Default parameters follow those provided in the original paper.
            # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
            opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

        else:
            raise Exception('Invalid Optimizer. Exiting')

        self.optimizer = opt

        # restriction about reguralization
        if kernel_regularization_params:
            assert kernel_regularization_params[0] in ['l1', 'l2']
            value = kernel_regularization_params[1]

            if kernel_regularization_params[0] == 'l1':
                self.kernel_regularizer = regularizers.l1(value)

            elif kernel_regularization_params[0] == 'l2':
                self.kernel_regularizer = regularizers.l2(value)

        if activity_regularization_params:
            assert activity_regularization_params[0] in ['l1', 'l2']
            value = activity_regularization_params[1]

            if activity_regularization_params[0] == 'l1':
                self.activity_regularizer = regularizers.l1(value)

            elif activity_regularization_params[0] == 'l2':
                self.activity_regularizer = regularizers.l2(value)

        # print("Number of Training examples = {}".format(self.n_examples_x_train))
        # print("Number of Test examples = {}".format(self.X_test.shape[0]))
        # print("Number of Features: {}".format(self.n_x_features))
        # print("Number of Classes: {}".format(self.n_y))
        # print("X_train shape: {}".format(X_train.shape))
        # print("Y_train shape: {}".format(Y_train.shape))
        # print("X_test shape: {}".format(X_test.shape))
        # print("Y_test shape: {}".format(Y_test.shape), end='\n\n')

    @staticmethod
    def vectorize_sequences(sequences, dimensions=10000):
        """

        :param sequences:
        :param dimensions:
        :return:
        """
        results = np.zeros((len(sequences), dimensions))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1

        return results

    @staticmethod
    def to_one_hot(labels, dimensions=46):
        """

        :param labels:
        :param dimensions:
        :return:
        """
        results = np.zeros(len(labels), dimensions)
        for i, label in enumerate(labels):
            results[i, label] = 1.

        return results

    def prepare_data(self, train_data, train_labels):
        """

        :param train_data:
        :param train_labels:
        :return:
        """
        self.train_data = train_data
        self._x_train = self.vectorize_sequences(train_data)

        self._x_val = self._x_train[:1000]
        self._partial_x_train = self._x_train[1000:]

        self._y_val = train_labels[:1000]
        self._partial_y_train = self._y_val[1000:]

    def build_model(self):
        pass

    def fit(self):
        """

        :return:
        """
        # self.prepare_data(self.train_data, self.train_labels)
        self.build_model()

        history = self.model.fit(self._partial_x_train,
                                 self._partial_y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self._x_val, self._y_val))

        return history
