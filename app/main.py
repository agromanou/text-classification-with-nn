import numpy as np

from app.models.simple_mlp import SimpleMLP

if __name__ == '__main__':
    train_data = np.array([[1., 2., 2., 1., 3.],
                           [1., 2., 2., 1., 3.],
                           [1., 2., 2., 1., 3.],
                           [1., 2., 2., 1., 3.],
                           [1., 2., 2., 1., 3.],
                           [1., 2., 2., 1., 3.]
                           ])
    train_labels = np.array([1, 2])

    print(train_data.shape)

    layers = list([50, 50, 50])
    loss = 'categorical_crossentropy'

    mlp = SimpleMLP(layers=layers,
                    loss=loss,
                    train_data=train_data,
                    train_labels=train_labels)

    mlp.build_model()
    mlp.fit()
