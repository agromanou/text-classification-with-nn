from app import MODELS_DIR

import os
import numpy as np
from matplotlib import pyplot as plt

from keras import callbacks
from keras import optimizers
from keras import regularizers

from keras.models import load_model as keras_load_model
from keras.utils import plot_model as keras_plot_model

plt.style.use('ggplot')


class ModelNN:
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
        :param loss: str, the name of the loss function
        :param optimizer: str, the name of the optimizer
        :param learning_rate: float, the learning rate
        :param decay: float, the decay
        :param momentum: float, momentum
        :param kernel_regularization_params: tuple, the regularization params
        :param epochs: int, the number of epochs
        :param batch_size: int, the size of the batch
        :param validation_size: float, the percentage of the validation split
        :param outfile:
        :param plot_model: boolean, if charts should plotted
        :param load_model: boolean, if existing trained model is available
        """

        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.plot_model = plot_model
        self.outfile = outfile
        self.load_model = load_model

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.load_trained_model()

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

    def train_dev_split(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        np.random.seed(200)

        # shuffling the training instances
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        train_data = X[indices]
        train_labels = y[indices]

        # calculating number of validation examples
        nb_validation_samples = int(self.validation_size * train_data.shape[0])

        # splitting in training and validation data
        x_train = train_data[:-nb_validation_samples]
        y_train = train_labels[:-nb_validation_samples]
        x_val = train_data[-nb_validation_samples:]
        y_val = train_labels[-nb_validation_samples:]

        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.y_val = y_val

        return {'x_train': x_train, 'x_val': x_val, 'y_train': y_train, 'y_val': y_val}

    def load_trained_model(self):
        if self.load_model and self.outfile:
            model_path = os.path.join(MODELS_DIR, self.outfile + '.h5')
            self.model = keras_load_model(model_path)

    def build_model(self, *kwargs):
        """
        Abstract method implements model building with keras
        """
        pass

    def fit(self, x_train, y_train):
        """

        :return:
        """
        print('EPOCHS: {}, BATCH SIZE: {}'.format(self.epochs, self.batch_size))

        tbCallBack = callbacks.TensorBoard(log_dir='./Graph',
                                           histogram_freq=0,
                                           write_graph=True,
                                           write_images=True)

        print('VALIDATION_SIZE: {}'.format(self.validation_size))

        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=self.validation_size,
                                 # validation_data=(self.X_val, self.y_val),
                                 verbose=2,
                                 callbacks=[tbCallBack])

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

        pred_scores = np.squeeze(predicted_classes)

        predicted_classes = list(map(lambda x: 1 if x > 0.5 else 0, list(pred_scores)))

        return {'scores': scores,
                'y_pred': predicted_classes,
                'y_pred_scores': pred_scores,
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
