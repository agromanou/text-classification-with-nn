from keras import models
from keras import layers
from keras import optimizers
import numpy as np


class Model:
    def __init__(self):
        self.model = None
        self.train_data = None

        self._x_train = None
        self._partial_x_train = None
        self._partial_y_train = None
        self._x_val = None
        self._y_val = None

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

    def fit(self, train_data, train_labels, epochs, batch_size):
        """

        :return:
        """
        self.prepare_data(train_data, train_labels)
        self.build_model()

        history = self.model.fit(self._partial_x_train,
                                 self._partial_y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self._x_val, self._y_val))

        return history


class SimpleMLP(Model):
    def __init__(self):
        Model.__init__(self)

        self.layers_num = None

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu',
                               input_shape=(self.train_data.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class SimpleRNN(Model):
    def __init__(self):
        Model.__init__(self)

        self.layers_num = None

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.Embedding(10000, 32))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class GRU(SimpleRNN):
    def __init__(self):
        Model.__init__(self)

        self.layers_num = None

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.GRU(64))
        model.add(layers.GRU(64))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class LSTM(SimpleRNN):
    def __init__(self):
        Model.__init__(self)

        self.layers_num = None

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.LSTM(64))
        model.add(layers.LSTM(64))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model


class CNN(Model):
    def __init__(self):
        Model.__init__(self)

        self.layers_num = None

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()
        model.add(layers.Embedding(10000, 32))

        for layer in self.layers_num:
            if layer == self.layers_num[:-1]:
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.GlobalMaxPooling1D(5))
            else:
                model.add(layers.Conv1D(32, 7, activation='relu'))
                model.add(layers.MaxPooling1D(5))

        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
