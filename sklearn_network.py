import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def read_x_from_file(file):
    data_1 = []
    data_2 = []
    with open(file, 'rb') as f:
        block = bytes(f.read(3))
        while block:
            x = block[0] + block[1] * 256 + (block[2] >> 4) * 65536  # transform data to number
            #data.append([(x >> i) & 0x01 for i in range(0, 20)])     # transform number as 20 input
            data_1.append(x >> 10)
            data_2.append(x & 0b00000000001111111111)
            block = bytes(f.read(3))
    return data_1, data_2


def read_y_from_file(file):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(1))
        while block:
            #data_x = [0, 0]
            #data_x[block[0]] = 1
            data.append(int.from_bytes(block, byteorder='big'))
            block = bytes(f.read(1))
    return data


def split_model(model):
    return model[['X1', 'X2']], model['Y']


input_data_1, input_data_2 = read_x_from_file("legenda/X.bin")
output_data1 = read_y_from_file("legenda/Y1.bin")
data_classification = {'X1': input_data_1, 'X2': input_data_2, 'Y': output_data1}
df = pd.DataFrame(data=data_classification)


def NaiveBayes(model, clf, test_size=0.2):
    x, y = split_model(model)

    x_train, x_test = train_test_split(x, test_size=test_size)
    y_train, y_test = train_test_split(y, test_size=test_size)

    clf.fit(x_train, y_train)

    print(clf)
    y_pred = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    report = classification_report(y_test, clf.predict(x_test))
    print(report)
    # Print results
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
          .format(
              x_test.shape[0],
            (y_test != y_pred).sum(),
            100 * (1-(y_test != y_pred).sum()/x_test.shape[0])))


h = .02
classifiers = [GaussianNB(), BernoulliNB(), LDA(n_components=1)]

for clf in classifiers:
    NaiveBayes(df, clf)


def x(x) -> str:
    return []
