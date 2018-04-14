import os

import keras
import numpy as np
from keras import regularizers
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, RMSprop, Adadelta
from keras.utils import plot_model as keras_plot_model
from matplotlib import pyplot as plt

from app import MODELS_DIR
from app.preprocessing import prepare_user_plus_vector_based_features

plt.style.use('ggplot')


def plot_model_metadata(history):
    """

    :param history:
    :return:
    """
    #  "Accuracy"
    # plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(ymax=1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    # plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # plt.tight_layout()

    plt.show()


class SequentialMLP:
    def __init__(self,
                 X_train,
                 Y_train,
                 X_test,
                 Y_test,
                 deep_layers=None,
                 learning_rate=0.0001,
                 num_epochs=1500,
                 minibatch_size=32,
                 decay=1e-6,
                 momentum=0.9,
                 deep_activation='relu',
                 activation='softmax',
                 optimizer='adam',
                 loss='mean_squared_error',
                 kernel_regularization_params=('l2', 0.01),
                 activity_regularization_params=None,
                 dropout=None,
                 outfile=None,
                 plot_model=False
                 ):
        """
        This class implements a sequential MLP using keras. It creates dynamically deep layers by simply
        setting the 'deep_layers' parameter as a list of ints.

        :param X_train: training set, of shape (number of training examples, features size)
        :param Y_train: test set, of shape (number of training examples, output size)
        :param X_test: training set, of shape (number of test examples, features size)
        :param Y_test: test set, of shape (output size)
        :param Y_test: Number of Classes
        :param deep_layers: list, A list of number of neurons fort each deep layer without the output layer.
        :param learning_rate: learning rate of the optimization
        :param num_epochs: number of epochs of the optimization loop
        :param minibatch_size: int, size of a minibatch (prefered power of 2, and smaller that 1024
        :param activation: activation function.
        :param optimizer: tensorflow optimizer, Optimizer for the calculation of the cost
        """

        # restrictions about the optimizers and the loss functions.
        assert optimizer in ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']

        assert loss in ['mean_squared_error', 'mean_absolute_error',
                        'mean_squared_logarithmic_error', 'squared_hinge',
                        'hinge', 'categorical_hinge', 'categorical_crossentropy',
                        'binary_crossentropy', 'cosine_proximity']

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        if deep_layers is None:
            deep_layers = [25, 12]

        (self.n_examples_x_train, self.n_x_features) = X_train.shape

        self.n_y = Y_train.shape[1]  # n_y : output size

        # all layers with the number of neurons in each layer.
        self.layers = deep_layers

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.activation = activation
        self.deep_activation = deep_activation
        self.optimizer = optimizer
        self.decay = decay
        self.momentum = momentum
        self.loss = loss
        self.dropout = dropout if dropout else None
        self.plot_model = plot_model
        self.outfile = outfile
        self.kernel_regularizer = None

        if kernel_regularization_params:
            assert kernel_regularization_params[0] in ['l1', 'l2']
            value = kernel_regularization_params[1]

            if kernel_regularization_params[0] == 'l1':
                self.kernel_regularizer = regularizers.l1(value)

            elif kernel_regularization_params[0] == 'l2':
                self.kernel_regularizer = regularizers.l2(value)

        self.activity_regularizer = None

        if activity_regularization_params:

            assert activity_regularization_params[0] in ['l1', 'l2']
            value = activity_regularization_params[1]

            if activity_regularization_params[0] == 'l1':
                self.activity_regularizer = regularizers.l1(value)

            elif activity_regularization_params[0] == 'l2':
                self.activity_regularizer = regularizers.l2(value)

        print("Number of Training examples = {}".format(self.n_examples_x_train))
        print("Number of Test examples = {}".format(self.X_test.shape[0]))
        print("Number of Features: {}".format(self.n_x_features))
        print("Number of Classes: {}".format(self.n_y))
        print("X_train shape: {}".format(X_train.shape))
        print("Y_train shape: {}".format(Y_train.shape))
        print("X_test shape: {}".format(X_test.shape))
        print("Y_test shape: {}".format(Y_test.shape), end='\n\n')

        self.nn_model = Sequential()

    def build_model(self):
        """

        :return:
        """

        # In the first layer, we must specify the expected input data shape
        self.nn_model.add(Dense(self.layers[0],
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=self.kernel_regularizer,
                                activity_regularizer=self.activity_regularizer,
                                input_dim=self.n_x_features))
        self.nn_model.add(BatchNormalization())
        self.nn_model.add(Activation(self.deep_activation))

        if self.dropout:
            self.nn_model.add(Dropout(self.dropout))

        for n_neurons in self.layers[1:]:

            self.nn_model.add(Dense(n_neurons,
                                    kernel_initializer='glorot_normal',
                                    kernel_regularizer=self.kernel_regularizer,
                                    activity_regularizer=self.activity_regularizer)
                              )

            self.nn_model.add(BatchNormalization())

            self.nn_model.add(Activation(self.deep_activation))

            if self.dropout:
                self.nn_model.add(Dropout(self.dropout))

        if self.n_y > 2:
            self.nn_model.add(Dense(self.n_y))
            self.nn_model.add(BatchNormalization())
            self.nn_model.add(Activation(activation='softmax'))

        elif self.n_y == 2:
            self.nn_model.add(Dense(1))
            self.nn_model.add(BatchNormalization())
            self.nn_model.add(Activation(activation='sigmoid'))

        else:
            raise Exception('No Activation Function Specified')

        return self.nn_model

    def fit(self, create_plots=True):
        """

        :param create_plots:
        :return:
        """

        if self.optimizer == 'sgd':
            opt = SGD(lr=self.learning_rate, decay=self.decay, momentum=0.9, nesterov=True)

        elif self.optimizer == 'rmsprop':
            opt = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

        elif self.optimizer == 'adagrad':
            # defaults: lr=0.01, epsilon=1e-08, decay=0.0
            opt = Adagrad(lr=self.learning_rate, epsilon=1e-08, decay=0.0)

        elif self.optimizer == 'adadelta':
            # defaults: lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0
            opt = Adadelta(lr=self.learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)

        elif self.optimizer == 'adam':
            # Default parameters follow those provided in the original paper.
            # lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
            opt = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        else:
            raise Exception('Invalid Optimizer. Exiting')

        self.build_model()

        self.nn_model.compile(loss=self.loss, optimizer=opt, metrics=['accuracy'])

        # plot_losses = PlotLosses()

        history = self.nn_model.fit(x=self.X_train,
                                    y=self.Y_train,
                                    epochs=self.num_epochs,
                                    batch_size=self.minibatch_size,
                                    validation_split=0.2,
                                    # callbacks=[plot_losses],
                                    verbose=2)

        test_score = self.nn_model.evaluate(x=self.X_test,
                                            y=self.Y_test,
                                            batch_size=self.minibatch_size,
                                            verbose=2)

        if self.outfile:
            model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
            self.nn_model.save(model_path)

        if self.outfile and self.plot_model:
            model_img_path = MODELS_DIR + self.outfile + '.png'
            keras_plot_model(self.nn_model, to_file=model_img_path, show_shapes=True, show_layer_names=True)

        if create_plots:
            plot_model_metadata(history)

        return test_score

    def predict_single(self, X):
        """

        :param X:
        :return:
        """

        classes = self.nn_model.predict(X, batch_size=1)

        return np.argmax(classes)


if __name__ == "__main__":
    meta_dict = prepare_user_plus_vector_based_features()

    print(meta_dict.keys())

    X_train = meta_dict['X_train']
    X_test = meta_dict['X_test']
    y_train = meta_dict['y_train']
    y_test = meta_dict['y_test']
    y_train_enc = meta_dict['y_train_enc']
    y_test_enc = meta_dict['y_test_enc']

    # y_train_one_hot = keras.utils.to_categorical(y_train_enc, num_classes=2)
    # y_test_one_hot = keras.utils.to_categorical(y_test_enc, num_classes=2)

    obj = SequentialMLP(X_train=X_train,
                        Y_train=y_train_enc,
                        X_test=X_test,
                        Y_test=y_test_enc,
                        deep_layers=[50],
                        learning_rate=0.0001,
                        num_epochs=1500,
                        minibatch_size=32,
                        deep_activation='relu',
                        activation='sigmoid',
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        kernel_regularization_params=('l2', 0.01),
                        dropout=0.2)

    obj.fit(create_plots=True)
