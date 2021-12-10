# from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
# import random
# import matplotlib.pyplot as plt
# import matplotlib
# import copy

# from gradient_check import check_layer_gradient, check_layer_param_gradient, check_model_gradient
from layers import FullyConnectedLayer, l2_regularization
from trainer import Trainer, Dataset
from optim import SGD, MomentumSGD
from metrics import r2_accuracy, mse, r2


def read_data(path="boston.csv"):
    dataframe = np.genfromtxt(path, delimiter=",", skip_header=15)
    np.random.seed(42)
    np.random.shuffle(dataframe)
    X = dataframe[:, :-1]
    y = dataframe[:, -1]
    return X, y


def generate_synthetic(size: int, dim=6, noise=0.1):
    X = np.random.randn(size, dim)
    w = np.random.randn(dim + 1)
    noise = noise * np.random.randn(size)
    y = X.dot(w[1:]) + w[0] + noise
    return X, y


class NormalLR:

    def __init__(self):
        self._weights = np.array = None
        self._x = np.array = None
        self._y = np.array = None

        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._x = X.copy()
        self._y = y.copy()
        # a = np.array( [[1 for i in range(X.shape[0])], [1 for i in range(X.shape[1]+1)]] )
        a = np.ones((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        a[:, :-1] = X
        x_T = np.transpose(a)
        self._weights = np.linalg.inv(x_T @ a) @ x_T @ self._y

        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        a = np.ones((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        a[:, :-1] = X

        if self._weights is None:
            RuntimeError("Error. regr wasn't fit")
        return a @ self._weights

class GradientLR:
    def __init__(self, n_input, n_output, alpha: float, iterations=10000, l=0.):
        self.model = FullyConnectedLayer(n_input=n_input, n_output=n_output);
        self.batch_size = 100
        self.num_epochs = int(iterations / self.batch_size)
        self.alpha = alpha
        self.reg = l

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params():
            self.params()[param].grad = np.zeros_like(self.params()[param].grad);

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        X1 = self.model.forward(X);

        loss, dX1 = mse(y, X1);
        dX = self.model.backward(dX1);

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params():
            if (param[0] == 'W'):
                loss_, grad_ = l2_regularization(self.params()[param].value, self.reg);
                self.params()[param].grad += grad_;
                loss += loss_;
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):
        train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, shuffle=True)
        dataset = Dataset(train_X, train_y, val_X, val_y);
        trainer = Trainer(self, dataset, SGD(), accuracy=r2, num_epochs=self.num_epochs,
                          batch_size=self.batch_size, learning_rate=self.alpha, learning_rate_decay=1.0);

        # TODO Implement missing pieces in Trainer.fit function
        # You should expect loss to go down and train and val accuracy go up for every epoch
        loss_history, train_history, val_history = trainer.fit()
        return loss_history, train_history, val_history

    def predict(self, X: np.ndarray):
        return self.model.forward(X)

    def params(self):
        return self.model.params()


def build_plot(X_train, y_train, X_test, y_test):
    xs = np.arange(0.0, 0.002, 0.00002)
    errors = []
    for x in xs:
        regr = GradientLR(X_train.shape[1], y_train.shape[1], 0.1, iterations=10000, l=x)
        regr.fit(X_train, y_train)
        errors.append(mse(y_test, regr.predict(X_test))[0])
    return xs, errors


if __name__ == '__main__':
    X, y = generate_synthetic(1024)
    y = np.reshape(y, (len(y), -1))
    X, X_val, y, y_val = train_test_split(X, y, train_size=0.9, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

    regr = GradientLR(X_train.shape[1], y_train.shape[1] if len(y_train.shape) > 1 else 1, 0.1, iterations=10000)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(f"MSE: {mse(y_test, y_pred)}, R2: {r2(y_test, y_pred)}")

    xs, errors = build_plot(X_train, y_train, X_val, y_val)

    X, y = read_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=False)

    regr = NormalLR()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_val)
    print(f"MSE: {mse(y_val, y_pred)}, R2: {r2(y_val, y_pred)}")

    y_train = np.reshape(y_train, (len(y_train), -1))
    y_val = np.reshape(y_val, (len(y_val), -1))

    build_plot(X_train, y_train, X_val, y_val)