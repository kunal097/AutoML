from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
# Create your views here.


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl

from sklearn.metrics import accuracy_score


class Model(object):

    def __init__(self, model=None, name=None, path=None, accuracy=None):
        self.model = model
        self.name = name
        self.path = path
        self.accuracy = accuracy

    def train(self, x, y):
        self.model.fit(x, y)

    def save(self):
        self.path = 'media/' + self.name + '.pkl'
        pkl.dump(self.model, open(self.path, 'wb'))

    def calculate_aaccuracy(self, y, y_pred):
        self.accuracy = accuracy_score(y, y_pred)
        return self.accuracy


class Preprocess(object):

    def __init__(self, data, label=-1, imputer_method='mean', normalize=True):
        self.data = pd.read_csv(data)
        self.label = int(label)
        self.imputer_method = imputer_method
        self.normalize = normalize

    def normalize_data(self):
        sc_X = StandardScaler()
        self.x = sc_X.fit_transform(self.x)
        sc_y = StandardScaler()
        self.y = sc_y.fit_transform(self.y)

    def handle_values(self):

        imputer = Imputer(missing_values='NaN',
                          strategy=self.imputer_method, axis=0)
        imputer = imputer.fit(self.x)
        self.x = imputer.transform(self.x)

    def process(self):
        self.x = self.data.iloc[:, :self.label]
        self.y = self.data.iloc[:, self.label]

        # if self.normalize:
        #     self.normalize_data()

        # self.handle_values()


class ModelTraining(object):
    """docstring for ClassName"""

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=20)

    def train_decision_tree(self):
        model = Model(DecisionTreeClassifier(), 'Decision Tree')
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()

        return score

    def train_logistic_regression(self):
        model = Model(LogisticRegression(), 'Logistic Regression')
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score

    def train_svm(self):
        model = Model(SVC(), 'SVM')
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score

    def train_random_forest(self):
        model = Model(RandomForestClassifier(), 'Random Forest')
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score


class GetData(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        print(request.data['file'])
        file = request.data['file']
        # algo_option = request.data['algo']
        normalize = request.data.get('normalize')
        label = request.data.get('label')
        imputer_method = request.data.get('imputer method')

        # print(file.read())

        path = default_storage.save(str(file), ContentFile(file.read()))

        print(path, '\n\n\n\n\n')
        # print(dir(request))
        path = 'media/' + path

        process_data = Preprocess(path)

        process_data.process()

        model = ModelTraining(process_data.x, process_data.y)

        acc1 = model.train_decision_tree()
        acc2 = model.train_random_forest()
        acc3 = model.train_logistic_regression()
        acc4 = model.train_svm()

        acc = [acc1,acc2,acc3,acc4]

        return Response({'Hello': acc})
