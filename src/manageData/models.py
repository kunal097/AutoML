from django.db import models
from django.contrib.auth.models import User

# Create your models here.


# user
# data
# settings
# model details

class Enduser(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    data_path = models.CharField(max_length=30)
    fields_dtype = models.TextField(max_length=200)
    settings = models.TextField(max_length=100)
    model_details = models.TextField(max_length=100)


    def __str__(self):
        return self.user



