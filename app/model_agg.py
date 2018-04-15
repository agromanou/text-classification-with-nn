from keras import optimizers
from keras import regularizers
from matplotlib import pyplot as plt

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
                 batch_size):

        self.model = None

        self.epochs = epochs
        self.batch_size = batch_size

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
            raise Exception('Invalid Optimizer. Exiting ', self.optimizer)

        self.optimizer = opt

        assert loss in ['mean_squared_error', 'mean_absolute_error',
                        'mean_squared_logarithmic_error', 'squared_hinge',
                        'hinge', 'categorical_hinge', 'categorical_crossentropy',
                        'binary_crossentropy', 'cosine_proximity']

        self.loss = loss

        # restriction about reguralization
        if kernel_regularization_params:
            assert kernel_regularization_params[0] in ['l1', 'l2']
            value = kernel_regularization_params[1]

            if kernel_regularization_params[0] == 'l1':
                self.kernel_regularizer = regularizers.l1(value)

            elif kernel_regularization_params[0] == 'l2':
                self.kernel_regularizer = regularizers.l2(value)

    def build_model(self, input_shape, labels_number):
        pass

    def fit(self, x_train, y_train):
        """

        :return:
        """
        input_shape = (x_train.shape[1],)

        self.build_model(input_shape=input_shape,
                         labels_number=2)

        print('PRINT {}, {}'.format(self.epochs, self.batch_size))
        history = self.model.fit(x=x_train,
                                 y=y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_split=0.2,
                                 verbose=2)

        return history

    def predict(self, x_test, y_test):
        """

        :return:
        """
        test_score = self.model.evaluate(x=x_test,
                                         y=y_test,
                                         batch_size=self.batch_size,
                                         verbose=2)

        return test_score

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

