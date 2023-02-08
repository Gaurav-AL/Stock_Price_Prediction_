from unittest.util import _MAX_LENGTH
from django.db import models

# Create your models here.

class Close(models.Model):
    closing_price = models.DecimalField(decimal_places=10,max_digits=4000)

class PredictedClose(models.Model):
    predicted_closing_price = models.DecimalField(decimal_places=10,max_digits=4000)
