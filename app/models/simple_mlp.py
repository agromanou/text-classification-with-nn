from keras import layers
from keras import models

from app.models.model import Model


class SimpleMLP(Model):
    def __init__(self,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 layers,
                 loss,
                 epochs=1500,
                 batch_size=32,
                 activation='softmax',
                 deep_activation='relu',
                 learning_rate=0.001,
                 decay=1e-6,
                 momentum=0.9,
                 optimizer='adam',
                 kernel_regularization_params=('l2', 0.01),
                 activity_regularization_params=None,
                 dropout=None):

        self.layers = layers
        self.batch_size = batch_size
        self.layers_num = len(layers)
        self.deep_activation = deep_activation
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.kernel_regularization_params = kernel_regularization_params
        self.dropout = dropout

        self.train_data = x_train
        self.train_labels = y_train

        assert loss in ['mean_squared_error', 'mean_absolute_error',
                        'mean_squared_logarithmic_error', 'squared_hinge',
                        'hinge', 'categorical_hinge', 'categorical_crossentropy',
                        'binary_crossentropy', 'cosine_proximity']

        Model.__init__(self,
                       optimizer,
                       learning_rate,
                       decay,
                       momentum,
                       kernel_regularization_params,
                       activity_regularization_params,
                       epochs,
                       batch_size,
                       x_train,
                       y_train,
                       x_test,
                       y_test)

    def build_model(self):
        """

        :return:
        """
        model = models.Sequential()

        # In the first layer, we must specify the expected input data shape
        model.add(layers.Dense(64,
                               activation=self.deep_activation,
                               input_shape=(self.train_data.shape[1],)))
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
        n_y = len(self.train_labels)
        if n_y > 2:
            model.add(layers.Dense(n_y))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation='softmax'))

        elif n_y == 2:
            model.add(layers.Dense(1))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation='sigmoid'))

        else:
            raise Exception('No Activation Function Specified')

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=['accuracy'])

        print(model.summary())
        self.model = model

