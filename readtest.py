from numpy import *

train_path = './a9a/a9a'
test_path = './a9a/a9a.t'


def sigmoid(x):
    return 1 / (1 + exp(-x))


def IRSL(train_x, train_y, eps=0.0001):
    n_samples, n_features = shape(train_x)
    w = mat(0.01 * ones(n_features, float)).T
    diag = mat(zeros([n_samples, n_samples]))
    print('start fitting...')
    while True:
        yp = sigmoid(train_x * w)
        error = yp - train_y
        for i in range(n_samples):
            diag[(i, i)] = yp[i] * (1 - yp[i])
        diag_inv = mat(zeros([n_samples, n_samples]))
        for i in range(n_samples):
            diag_inv[(i, i)] = 1 / diag[(i, i)]
        H = train_x.T * diag * train_x
        print(H)
        # wn = H.I * train_x.T * diag * ( train_x * w - diag.I * error)
        # sub_error = wn - w
        # max_error = 0
        # for e in sub_error.flat:
        #     max_error = max(max_error, abs(e))
        # if max_error < eps:
        #     break
        # w = wn
    print('fitting success!')
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
    target_list = []
    data_list = []
    for line in open(train_path).readlines():
        meta = line.split(' ')
        target_list.append(float(meta[0]))
        loc = zeros(128)
        for c in meta[1:]:
            try:
                loc[int(c.split(':')[0])] = 1
            except ValueError:
                pass
        data_list.append(loc)
    return mat(data_list), mat(target_list).transpose()


def main():
    (train_data, train_target) = load_data(train_path)
    (test_data, test_target) = load_data(test_path)
    w = IRSL(train_data, train_target)
    # raise ('w with IRLS is : ' + w.tostring())
    # ac = accuracy(test_data, test_target, w)
    # raise ('accuracy is : ' + string(ac))


if __name__ == '__main__':
    main()
