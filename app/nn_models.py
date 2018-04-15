from keras import layers
from keras import models

from app.models.model import Model


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
