import itertools as it

from keras import layers
from keras import models

from app.model import ModelAgg
from app.preprocessing import prepare_user_plus_vector_based_features


class SimpleMLP(ModelAgg):
    def __init__(self,
                 layers_structure,
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

        ModelAgg.__init__(self,
                          optimizer,
                          learning_rate,
                          decay,
                          momentum,
                          kernel_regularization_params,
                          activity_regularization_params,
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
    meta_dict = prepare_user_plus_vector_based_features()

    print(meta_dict.keys())

    X_train = meta_dict['X_train']
    X_test = meta_dict['X_test']
    y_train = meta_dict['y_train']
    y_test = meta_dict['y_test']
    y_test_enc = meta_dict['y_test_enc']

    params = {'deep_layers': [[20, 20, 20],
                              [40, 60, 40],
                              [30, 30, 30]],
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

    layers_structure = [20, 20, 20]
    loss = 'binary_crossentropy'

    mlp = SimpleMLP(layers_structure,
                    loss)

    # mlp.build_model()
    # mlp.fit()
