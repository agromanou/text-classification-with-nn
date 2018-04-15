import itertools as it
import operator
from pprint import pprint

import numpy as np
from keras import layers
from keras import models

from app.model_agg import Model
from app.preprocessing import prepare_user_plus_vector_based_features


class SimpleMLP(Model):
    def __init__(self,
                 layers_structure,
                 loss,
                 epochs=100,
                 batch_size=32,
                 activation='softmax',
                 deep_activation='relu',
                 learning_rate=0.001,
                 decay=1e-6,
                 momentum=0.9,
                 optimizer='adam',
                 kernel_regularization_params=('l2', 0.01),
                 dropout=0.3):

        self.layers = layers_structure
        self.batch_size = batch_size
        self.layers_num = len(layers_structure)
        self.deep_activation = deep_activation
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.kernel_regularization_params = kernel_regularization_params
        self.dropout = dropout

        Model.__init__(self,
                       loss,
                       optimizer,
                       learning_rate,
                       decay,
                       momentum,
                       kernel_regularization_params,
                       epochs,
                       batch_size)

    def build_model(self, input_shape, labels_number):
        """
        Creates and compiles an mlp model with Keras
        :param input_shape: tuple, the shape of the input layer
        :param labels_number: int, tha number of categorical values of labels
        """
        model = models.Sequential()

        # In the first layer, we must specify the expected input data shape
        model.add(layers.Dense(64,
                               activation=self.deep_activation,
                               input_shape=input_shape))
        if self.dropout:
            model.add(layers.Dropout(self.dropout))

        # add hidden layers
        for n_neurons in self.layers[1:]:
            model.add(layers.Dense(n_neurons,
                                   activation=self.deep_activation,
                                   kernel_regularizer=self.kernel_regularizer))
            model.add(layers.BatchNormalization())

            model.add(layers.Activation(self.deep_activation))

            if self.dropout:
                model.add(layers.Dropout(self.dropout))

        # add last layer
        if labels_number == 2:
            model.add(layers.Dense(1))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation='sigmoid'))

        elif labels_number > 2:
            model.add(layers.Dense(labels_number))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation='softmax'))

        else:
            raise Exception('No Activation Function Specified')

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=['accuracy'])

        print(model.summary())
        self.model = model


if __name__ == '__main__':
    data = prepare_user_plus_vector_based_features()

    x_train = data['X_train']
    y_train = data['y_train']
    x_test = data['X_test']
    y_test = data['y_test']
    y_train_enc = data['y_train_enc']
    y_test_enc = data['y_test_enc']

    (n_examples_x_train, n_x_features) = x_train.shape

    n_y = y_train_enc.shape[1]  # n_y : output size

    print("Number of Training examples = {}".format(n_examples_x_train))
    print("Number of Test examples = {}".format(x_test.shape[0]))
    print("Number of Features: {}".format(n_x_features))
    print("Number of Classes: {}".format(n_y))
    print("X_train shape: {}".format(x_train.shape))
    print("Y_train shape: {}".format(y_train_enc.shape))
    print("X_test shape: {}".format(x_test.shape))
    print("Y_test shape: {}".format(y_test_enc.shape), end='\n\n')

    params = {'deep_layers': [(20, 20, 20),
                              (40, 60, 40),
                              (30, 30, 30)],
              'learning_rate': [0.001, 0.01],
              'optimizer': ['rmsprop', 'adam', 'sgd'],
              'loss': ['binary_crossentropy'],
              'deep_activation': ['relu', 'tanh'],
              'activation': ['sigmoid']}

    comb = it.product(params['deep_layers'],
                      params['learning_rate'],
                      params['optimizer'],
                      params['loss'],
                      params['deep_activation'],
                      params['activation'])

    average = dict()
    histories = dict()
    with open('results_mlp.txt', 'a') as f:
        for i in comb:
            print(i)
            mlp = SimpleMLP(layers_structure=i[0],
                            learning_rate=i[1],
                            optimizer=i[2],
                            loss=i[3],
                            deep_activation=i[4],
                            activation=i[5])

            history = mlp.fit(x_train=x_train,
                              y_train=y_train_enc)

            average[tuple(i)] = np.mean(history.history['val_acc'][40:])
            histories[tuple(i)] = {'acc': history.history['acc'],
                                   'acc_val': history.history['val_acc']}

            print('-' * 30, 'END OF RUN', '-' * 30)

            for i in range(0, len(history), 10):
                f.write(str(i) + ', ' + str(history[i]) + '\n')

    pprint(average)

    best_settings = max(average.items(), key=operator.itemgetter(1))[0]

    mlp_best = SimpleMLP(layers_structure=best_settings[0],
                         learning_rate=best_settings[1],
                         optimizer=best_settings[2],
                         loss=best_settings[3],
                         deep_activation=best_settings[4],
                         activation=best_settings[5])

    history = mlp_best.fit(x_train=x_train,
                           y_train=y_train_enc)

    mlp_best.plot_model_metadata(history)

    test_score = mlp_best.predict(x_test=x_test, y_test=y_test_enc)

    print(test_score)
