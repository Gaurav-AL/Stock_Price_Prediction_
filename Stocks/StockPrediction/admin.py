from django.contrib import admin
from .models import Close,PredictedClose
# Register your models here.

admin.site.register(Close)
admin.site.register(PredictedClose)
