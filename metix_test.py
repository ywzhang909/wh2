from numpy import *


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def IRSL(x, y, sigma=0.0001):
    n_samples, n_features = shape(x)
    w = ones(n_features, 0.01)
    Rnn = mat(zeros(n_samples, n_samples))
    while True:
        _y = sigmoid(x * w)
        error = _y - y
        for i in range(n_samples):
            Rnn[i, i] = _y[i] * (1 - _y[i])
        H = x.T * Rnn * x.T
        _w = H.I * x.T * Rnn * (x * w - Rnn.I * error)
        if max(abs(w - _w)) < sigma:
            break
        w = _w
    return w


def main():
    x = array([1, 2, 3, 4, 5])
    print(sigmoid(x))


if __name__ == '__main__':
    main()
