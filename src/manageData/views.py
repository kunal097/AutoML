from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
# Create your views here.


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.metrics import accuracy_score


class ModelTraining(object):
    """docstring for ClassName"""
    def __init__(self, path):
        self.path = path
        self.model = DecisionTreeClassifier()
        self.data = pd.read_csv(path)

    def train(self):

        X_train, X_test, y_train, y_test = train_test_split(self.data.iloc[:,:-1], self.data.iloc[:,-1], test_size=0.2, random_state=20)

        self.model.fit(X_train, y_train)

        predicted_result = self.model.predict(X_test)
        score = accuracy_score(y_test, predicted_result)

        model_name = '/'.join(self.path.split('/')[:-1])+'/model.pkl'


        pkl.dump(self.model, open(model_name, 'wb'))


        print(score)



class GetData(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        print(request.data['file'])
        file = request.data['file']
        algo_option = request.data['algo']

        # print(file.read())





        path = default_storage.save(str(file), ContentFile(file.read()))

        print(path,'\n\n\n\n\n')
        # print(dir(request))
        path = 'media/'+path

        model = ModelTraining(path)

        model.train()


        return Response({'Hello': [path, algo_option]})



