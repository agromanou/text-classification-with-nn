import os

import numpy as np
from keras import optimizers
from keras import regularizers
from keras.models import load_model
from keras.utils import plot_model as keras_plot_model
from matplotlib import pyplot as plt

from app import MODELS_DIR

plt.style.use('ggplot')


class Model:
    def __init__(self,
                 loss,
                 optimizer,
                 learning_rate,
                 decay,
                 momentum,
                 kernel_regularization_params,
                 epochs,
                 batch_size,
                 validation_size,
                 outfile=None,
                 plot_model=False,
                 load_model=False
                 ):
        """

        :param loss:
        :param optimizer:
        :param learning_rate:
        :param decay:
        :param momentum:
        :param kernel_regularization_params:
        :param validation_size:
        :param epochs:
        :param batch_size:
        """

        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.plot_model = plot_model
        self.outfile = outfile
        self.load_model = load_model

        # restrictions about the optimizers
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
            raise Exception('Invalid Optimizer. Existing: ', self.optimizer)
        self.optimizer = opt

        assert loss in ['mean_squared_error', 'mean_absolute_error',
                        'mean_squared_logarithmic_error', 'squared_hinge',
                        'hinge', 'categorical_hinge', 'categorical_crossentropy',
                        'binary_crossentropy', 'cosine_proximity']

        self.loss = loss

        # restriction about regularization
        if kernel_regularization_params:
            assert kernel_regularization_params[0] in ['l1', 'l2']
            value = kernel_regularization_params[1]

            if kernel_regularization_params[0] == 'l1':
                self.kernel_regularizer = regularizers.l1(value)

            elif kernel_regularization_params[0] == 'l2':
                self.kernel_regularizer = regularizers.l2(value)

    def load_model(self):
        if self.load_model and self.outfile:
            model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
            self.model = load_model(model_path)

    def build_model(self, input_shape, labels_number):
        """
        Abstract method implements model building with keras
        """
        pass

    def fit(self, x_train, y_train):
        """

        :return:
        """
        input_shape = (x_train.shape[1],)

        self.build_model(input_shape=input_shape,
                         labels_number=2)

        print('EPOCHS: {}, BATCH SIZE: {}'.format(self.epochs, self.batch_size))

        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=self.validation_size,
                                 verbose=2)

        if self.outfile:
            model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
            self.model.save(model_path)

        if self.outfile and self.plot_model:
            model_img_path = MODELS_DIR + self.outfile + '.png'
            keras_plot_model(self.model,
                             to_file=model_img_path,
                             show_shapes=True,
                             show_layer_names=True)

        return history

    def predict(self, X, y):
        """

        :return:
        """
        scores = self.model.evaluate(x=X,
                                     y=y,
                                     batch_size=self.batch_size,
                                     verbose=2)

        predicted_classes = self.model.predict(X, batch_size=self.batch_size)
        predicted_classes = list(map(lambda x: 1 if x > 0.5 else 0, list(np.squeeze(predicted_classes))))

        return {'scores': scores,
                'y_pred': predicted_classes,
                'y_true': y}

    @staticmethod
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
        plt.show()
