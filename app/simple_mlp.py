from app.evaluation import create_clf_report
from app.model import ModelNN
from app.preprocessing import prepare_user_plus_vector_based_features
from app.evaluation import plot_roc_curve, plot_precision_recall_curve

import itertools as it
import operator
import numpy as np

from keras import layers
from keras import models


class SequentialMLP(ModelNN):
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
                 dropout=0.3,
                 validation_size=0.2,
                 outfile=None,
                 plot_model=False,
                 load_model=False
                 ):
        """
        :param layers_structure: list of int, with the structure of the hidden layers
        :param loss: str, the name of the loss function
        :param epochs: int, the number of epochs
        :param batch_size: int, the size of the batch
        :param activation: str, the name of the activation function
        :param deep_activation: str, the name of the activation function in hidden layers
        :param learning_rate: float, the learning rate
        :param decay: float, the decay
        :param momentum: float, momentum
        :param optimizer: str, the name of the optimizer
        :param kernel_regularization_params: tuple, the regularization params
        :param dropout: float, dropout probability
        :param validation_size: float, the percentage of the validation split
        :param outfile:
        :param plot_model: boolean, if charts should plotted
        :param load_model: boolean, if existing trained model is available
        """

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
        self.outfile = outfile
        self.plot_model = plot_model
        self.load_model = load_model

        ModelNN.__init__(self,
                         loss=loss,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         decay=decay,
                         momentum=momentum,
                         kernel_regularization_params=kernel_regularization_params,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_size=validation_size,
                         outfile=outfile,
                         plot_model=plot_model,
                         load_model=load_model)

    def build_model(self, input_shape, labels_number):
        """
        Creates and compiles an mlp model with Keras deep learning library
        :param input_shape: tuple, the shape of the input layer
        :param labels_number: int, tha number of categorical values of labels
        """
        model = models.Sequential()

        # In the first layer, we must specify the expected input data shape
        model.add(layers.Dense(64,
                               kernel_initializer='glorot_normal',
                               activation=self.deep_activation,
                               kernel_regularizer=self.kernel_regularizer,
                               input_shape=input_shape))
        if self.dropout:
            model.add(layers.Dropout(self.dropout))

        # add hidden layers
        for n_neurons in self.layers[1:]:
            model.add(layers.Dense(n_neurons,
                                   kernel_initializer='glorot_normal',
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


def run_parameter_tuning():
    """
    This function runs the hyper-parameter tuning and plots the learning curves for the best model found.
    """
    data = prepare_user_plus_vector_based_features()

    x_train = data['X_train']
    y_train_enc = data['y_train_enc']
    input_shape = (x_train.shape[1],)

    (n_examples_x_train, n_x_features) = x_train.shape

    n_y = y_train_enc.shape[1]  # n_y : output size

    print("Number of Training examples = {}".format(n_examples_x_train))
    print("Number of Features: {}".format(n_x_features))
    print("Number of Classes: {}".format(n_y))
    print("X_train shape: {}".format(x_train.shape))
    print("Y_train shape: {}".format(y_train_enc.shape), end='\n\n')

    params = {'deep_layers': [(20,),
                              (60,),
                              (100,),
                              (20, 20, 20),
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
            mlp = SequentialMLP(layers_structure=i[0],
                                learning_rate=i[1],
                                optimizer=i[2],
                                epochs=500,
                                loss=i[3],
                                deep_activation=i[4],
                                activation=i[5])

            mlp.build_model(input_shape=input_shape, labels_number=2)

            history = mlp.fit(x_train=x_train,
                              y_train=y_train_enc)

            average[tuple(i)] = np.mean(history.history['val_acc'][40:])
            histories[tuple(i)] = {'acc': history.history['acc'],
                                   'acc_val': history.history['val_acc']}

            print('-' * 30, 'END OF RUN', '-' * 30)

            for n in range(0, len(history.history['val_acc']), 10):
                f.write(str(n) + ', ' + str(history.history['val_acc'][n]) + '\n')
            break

    print(average)

    best_settings = max(average.items(), key=operator.itemgetter(1))[0]

    mlp_best = SequentialMLP(layers_structure=best_settings[0],
                             learning_rate=best_settings[1],
                             optimizer=best_settings[2],
                             loss=best_settings[3],
                             epochs=500,
                             deep_activation=best_settings[4],
                             activation=best_settings[5])

    mlp_best.build_model(input_shape=input_shape, labels_number=2)

    history = mlp_best.fit(x_train=x_train, y_train=y_train_enc)

    mlp_best.plot_model_metadata(history)


def best_mpl_model(load_pre_trained=True):
    """
    Test the best model on test data and plot resutls
    :param load_pre_trained: boolean, check if a model is already pre_trained
    in order to skip training and perform evaluation
    """
    data = prepare_user_plus_vector_based_features()

    x_train = data['X_train']
    x_test = data['X_test']
    y_train = data['y_train_enc']
    y_test = data['y_test_enc']

    input_shape = (x_train.shape[1],)

    (n_examples_x_train, n_x_features) = x_train.shape

    n_y = y_train.shape[1]  # n_y : output size

    print("Number of Training examples = {}".format(n_examples_x_train))
    print("Number of Test examples = {}".format(x_test.shape[0]))
    print("Number of Features: {}".format(n_x_features))
    print("Number of Classes: {}".format(n_y))
    print("X_train shape: {}".format(x_train.shape))
    print("Y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(x_test.shape))
    print("Y_test shape: {}".format(y_test.shape), end='\n\n')

    mlp_best = SequentialMLP(layers_structure=[20],
                             learning_rate=0.001,
                             optimizer='sgd',
                             loss='binary_crossentropy',
                             epochs=200,
                             deep_activation='tanh',
                             activation='sigmoid',
                             validation_size=0,
                             outfile='best_seq_mlp_model',
                             plot_model=False,
                             load_model=load_pre_trained)

    mlp_best.build_model(input_shape=input_shape, labels_number=2)

    if not load_pre_trained:
        mlp_best.fit(x_train=x_train, y_train=y_train)

    meta = mlp_best.predict(X=x_test, y=y_test)

    print("Loss: {}, Acc: {}".format(meta['scores'][0], meta['scores'][1]))

    y_true = meta['y_true']
    y_pred = meta['y_pred']
    y_pred_scores = meta['y_pred_scores']

    create_clf_report(y_true, y_pred, ['negative', 'positive'])
    plot_roc_curve(y_true, y_pred_scores, pos_label=1)
    plot_precision_recall_curve(y_true, y_pred_scores, pos_label=1)


if __name__ == '__main__':
    run_parameter_tuning()
    # best_mpl_model(load_pre_trained=True)
