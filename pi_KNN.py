from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import random
import pandas as pd

X, y = datasets.make_classification(n_samples=600, n_classes=3,
                                    n_features=10, n_informative=3)

X_train, X_test, y_train, y_test = train_test_split(X, y)

class KNN(object):
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

# класс оценки важности признаков (permutation importance)
class PermutationImportance(object):
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.cv = cv
        self.importance = []
        self.f_score = []

    def evaluate_it(self):
        original_score = self.model.cv(X, y, cv=self.cv).mean()

        for i in range(X.shape[1]):
            X_new = X.copy()
            np.random.shuffle(X_new[:, i])

            feature_score = self.model.cv(X_new, y, cv=self.cv).mean()
            self.f_score.append(feature_score)
            self.importance.append(feature_score - original_score)

        return self.importance

CV = 3
clf = KNN()
cv = clf.cv(X, y, cv=CV)

print("Cross validation score:", cv, " Its mean:", str(cv.mean())[:4])
print("Starting evaluate feature importance...\n")

imp = PermutationImportance(clf, X, y, cv=CV)
imp_per_feature = imp.evaluate_it()

print("Feature importance by Permutation:")

for i, score in enumerate(imp_per_feature):
    if score < 0:
        print("feature", i, "its importance", str(score * (-1.0))[:4])



