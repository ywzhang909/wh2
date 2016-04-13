from numpy import *


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def fit(train_x, train_y, alpha=0.01, max_iter=10, optimize_method='IRLS'):
    n_samples, n_features = shape(train_x)
    w = ones(n_features, 1)
    Rnn = mat(zeros((n_samples, n_samples)))
    for k in max_iter:
        if optimize_method == 'gradDescent':
            y = sigmoid(train_x * w)
            error = train_y - y
            w += alpha * train_x.transpose() * error
        elif optimize_method == 'stocGradDecent':
            for i in range(n_samples):
                y = sigmoid(train_x[i, :] * w)
                error = train_y - y
                w += alpha * train_x[i, :].transpose() * error
        elif optimize_method == 'IRLS':
            y = sigmoid(train_x * w)
            error = train_y - y
            for i in range(n_samples):
                Rnn[i, i] = y[i] * (1 - y[i])
            H = train_x.transpose() * Rnn * train_x.transpose()
            w = H.I * train_x.T * Rnn * (train_x * w - Rnn.I * error)
        else:
            raise NameError("Not support optimize method type!")
    print("fit success!")
    return w


def accuracy(test_x, test_y, w):
    n_samples, n_features = shape(test_x)
    match_count = 0
    for i in range(n_samples):
        predict = sigmoid(test_x[i, :] * w)[0, 0] > 0.5
        if predict == (test_y[i, :] == 1):
            match_count += 1
    return float(match_count) / n_samples


def load_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        meta = line.split(' ')
        y.append(float(meta[0]))
        feature = zeros(128)
        for c in meta[1:]:
            try:
                feature[int(c.split(':')[0])] = 1
            except ValueError:
                raise 'ValueError'
        x.append(feature)
    return mat(x), mat(y).T


def main():
    print(accuracy(load_data('./a9a/a9a.t'), fit(load_data('./a9a/a9a'))))


if __name__ == '__main':
    main()





