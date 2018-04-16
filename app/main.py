from app.simple_mlp_old import SimpleMLP

from app.preprocessing import prepare_user_plus_vector_based_features

if __name__ == '__main__':
    meta_dict = prepare_user_plus_vector_based_features()

    print(meta_dict.keys())

    X_train = meta_dict['X_train']
    X_test = meta_dict['X_test']
    y_train = meta_dict['y_train']
    y_test = meta_dict['y_test']
    y_train_enc = meta_dict['y_train_enc']
    y_test_enc = meta_dict['y_test_enc']

    print(X_train.shape)

    layers = list([50, 50, 50])
    loss = 'categorical_crossentropy'

    mlp = SimpleMLP(layers=layers,
                    loss=loss,
                    x_train=X_train,
                    y_train=y_train,
                    x_test=X_test,
                    y_test=y_test)

    mlp.build_model()
    mlp.fit()
