from django.contrib import admin
from django.urls import path,include
from StockPrediction import views

urlpatterns = [
    path('index',views.index,name = "index"),
    path('getSymbol',views.getSymbol, name="getSymbol"),
]