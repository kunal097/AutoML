from django.urls import path
from manageData import views

urlpatterns = [
 path('',views.GetData.as_view())
]
