from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import random
import pandas as pd

X, y = datasets.make_classification(n_samples=500, n_classes=3,
                                    n_features=10, n_informative=5)

X_train, X_test, y_train, y_test = train_test_split(X, y)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.task = None
        self.last_indexes = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.find_task_type(y_train)

    def find_task_type(self, y_train):
        # метод распознавания типа решаемой задачи - классификация или регрессия
        self.task = "regression"
        if (y_train == y_train.astype(int)).sum() / len(y_train):
            self.task = "classification"

    def distance(self, x_test, x_train):
        return np.sqrt(np.sum((x_test - x_train) ** 2))

    def predict(self, X_test):
        labels = [self.find_labels(x_test) for x_test in X_test]
        return np.array(labels)

    def find_labels(self, x_test):
        distances = [self.distance(x_test, x_train) for x_train in self.X_train]
        k_nearest = np.argsort(distances)[:self.k]
        self.last_indexes = k_nearest
        k_labels = [self.y_train[i] for i in k_nearest]

        if self.task == "regression":
            return np.sum(k_labels) / self.k

        return self.most_common(k_labels)

    def most_common(self, k_labels):
        a = tuple(set(k_labels))
        most_common = [k_labels.count(i) for i in a]
        index = np.argsort(most_common)[-1]

        if len(set(most_common)) == 1 or len(most_common) == len(k_labels):
            return random.choice(a)

        return a[index]

    # методы проверки качества моделей для обоих типов задач
    def score(self, predicted, y_test):
        return (predicted == y_test).sum() / len(y_test)

    def r2(self, predicted, y_test):
        return 1 - np.sum((predicted - y_test) ** 2) / np.sum(
            (y_test.mean() - y_test) ** 2)

    # встроенный метод скользящего контроля (кросс валидация по фолдам)
    def cv(self, X, y, cv=5):
        self.find_task_type(y)

        y = np.reshape(y, (len(y), 1))
        data = np.concatenate((X, y), axis=1)
        np.random.shuffle(data)

        data = pd.DataFrame(data)
        score = []

        for i in range(cv):
            lenght = int(len(y) / cv)

            X_test = data.iloc[i * lenght:i * lenght + lenght, :-1]
            X_train = data.drop(index=X_test.index).iloc[:, :-1]

            y_test = data.iloc[i * lenght:i * lenght + lenght, -1]
            y_train = data.drop(index=X_test.index).iloc[:, -1]

            clf = KNN()
            clf.fit(np.array(X_train), np.array(y_train))
            predicted = clf.predict(np.array(X_test))

            if self.task == "classification":
                score.append(clf.score(predicted, np.array(y_test)))
            else:
                score.append(clf.r2(predicted, np.array(y_test)))

        return np.array(score)

# метод оценки важности признаков (permutation importance)
# def feature_importance(self, X_res, y_res, cv=5):
#     clf = KNN()
#     original = clf.cv(X_res, y_res, cv=cv, disable=True).mean()
#
#     importance = []
#
#     for i in tq(range(X_res.shape[1]), desc="Оценка важности признаков"):
#         X_new = X_res.copy()
#         np.random.shuffle(X_new[:, i])
#
#         cv = clf.cv(X_new, y_res, cv=5, disable=True)
#         importance.append(np.mean(cv))
#
#     s = int(np.sqrt(len(importance)))
#     importance = np.array(importance - original).reshape(s, s)
#
#     fig, axs = plt.subplots(figsize=(3, 3))
#     sns.heatmap(importance, annot=True, vmin=0, fmt='0.2f', ax=axs, cbar=False)
#     plt.show()

clf = KNN()
clf.fit(X_train, y_train)
cv = clf.cv(X, y)

print("Cross validation score:", cv, " Mean:", cv.mean())

