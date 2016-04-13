from sklearn.linear_model import LogisticRegression


def get_data(file_path):
    target_list = []
    data_list = []
    for line in open(file_path).readlines():
        meta = line.split(' ')
        target_list.append(int(meta[0]))
        loc = []
        for c in meta[1:]:
            try:
                loc.append(int(c.split(':')[0]))
            except ValueError:
                print(c.split(':')[0])
        data_list.append(loc)
    


def main():
    (train_target, train_data) = get_data('./a9a/a9a')
    (test_target, test_data) = get_data('./a9a/a9a.t')
    classifier = LogisticRegression()
    classifier.fit(train_data, train_target)


if __name__ == '__main__':
    main()
