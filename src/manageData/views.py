from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
# from django.core.files.storage import FileSystemStorage, Storage
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
from random import randint
from AutoML.settings import MEDIA_ROOT
import os

# MEDIA_ROOT = os.path.join(MEDIA_ROOT,'test')

# class User(object):

#     def __init__(self, path):
#         self.working_dir = path


class Model(object):

    def __init__(self, model=None, name=None, path=None, accuracy=None):
        self.model = model
        self.name = name
        self.path = path + '/' + self.name + '.pkl'
        self.accuracy = accuracy

    def train(self, x, y):
        self.model.fit(x, y)

    def save(self):
        # self.path = 'media/' + self.name + '.pkl'
        print(self.path)
        print("HHDHD(*D*D&D&DD^D^^D^D\n\n\n")
        pkl.dump(self.model, open(self.path, 'wb'))

    def calculate_aaccuracy(self, y, y_pred):
        self.accuracy = accuracy_score(y, y_pred)
        return self.accuracy


class Preprocess(object):

    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.label = -1
        # self.imputer_method = imputer_method
        # self.normalize = normalize

    def normalize_data(self):
        sc_X = StandardScaler()
        self.x = sc_X.fit_transform(self.x)
        # sc_y = StandardScaler()
        # self.y = sc_y.fit_transform(self.y)

    def apply_imputer(self):

        imputer = Imputer(missing_values='NaN',
                          strategy='mean', axis=0)
        imputer = imputer.fit(self.x)
        self.x = imputer.transform(self.x)

    # def process(self):
    #     self.x = self.data.iloc[:, :self.label]
    #     self.y = self.data.iloc[:, self.label]

        # if self.normalize:
        #     self.normalize_data()

        # self.handle_values()

    def check_nan(self):
        nan_list = self.data.isna().sum().to_list()
        max_nan = max(nan_list)
        rows = self.data.shape[0]

        return (( max_nan / rows ) * 100)

    def check_dtype(self):

        random_index = [randint(0, len(self.y)) for i in range(10)]
        dtype_list = [type(self.y[i]) == str  for i in random_index]

        if dtype_list.count(True) > dtype_list.count(False):
            return True

        return False





    def process(self):

        # calculate NaN ratio
        nan_ratio = self.check_nan()

        if nan_ratio <= 5:
            # drop rows which contain missing values
            self.data = self.data.dropna()
            self.x = self.data.iloc[:, :self.label]
            self.y = self.data.iloc[:, self.label]
        else:
            # Applying Imputer
            self.x = self.data.iloc[:, :self.label]
            self.y = self.data.iloc[:, self.label]
            self.apply_imputer()

        # check label datatype
        is_str = self.check_dtype()

        if not is_str:
            # Normalize data
            self.normalize_data()


class ModelTraining(object):
    """docstring for ClassName"""

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=20)

    def train_decision_tree(self, data_path):
        model = Model(DecisionTreeClassifier(), 'Decision Tree', data_path)
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()

        return score

    def train_logistic_regression(self, data_path):
        model = Model(LogisticRegression(), 'Logistic Regression', data_path)
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score

    def train_svm(self, data_path):
        model = Model(SVC(), 'SVM', data_path)
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score

    def train_random_forest(self, data_path):
        model = Model(RandomForestClassifier(), 'Random Forest', data_path)
        model.train(self.X_train, self.y_train)
        score = model.calculate_aaccuracy(
            self.y_test, model.model.predict(self.X_test))
        model.save()
        return score



class FeatureAnalysis(object):

    def __init__(self):
        pass


    def feature_selection(self):
        pass




class GetData(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        # print(request.data['file'])
        file = request.data['file']
        # print(dir(file))
        # print(dir(request))
        # print(request.META.get('REMOTE_ADDR'))
        # data_dir = request.META.get('REMOTE_ADDR')
        # print(tpath,MEDIA_ROOT)
        # print(os.path.join(MEDIA_ROOT,tpath))
        data_path = os.path.join(MEDIA_ROOT,request.META.get('REMOTE_ADDR'))

        # user = User(data_path)
        # print(data_path)
        try:
            os.mkdir(data_path)
        except:
            pass
        # print(os.path.join(f,str(file)))
        file_path = os.path.join(data_path, str(file))


        # f = open(file)
        # print(f.read())
        # f.close()
        # algo_option = request.data['algo']
        # normalize = request.data.get('normalize')
        # label = request.data.get('label')
        # imputer_method = request.data.get('imputer method')

        # print(ContentFile(file.read()))

        path = default_storage.save(file_path, ContentFile(file.read()))

        # print(path, '\n\n\n\n\n')
        # print(dir(request))
        # path = 'media/' + path
        # print(path)

        process_data = Preprocess(path)

        process_data.process()

        model = ModelTraining(process_data.x, process_data.y)
        # print(process_data.y)

        acc1 = model.train_decision_tree(data_path)
        acc2 = model.train_random_forest(data_path)
        acc3 = model.train_logistic_regression(data_path)
        acc4 = model.train_svm(data_path)

        acc = [acc1,acc2,acc3,acc4]

        return Response({'Hello': acc})
