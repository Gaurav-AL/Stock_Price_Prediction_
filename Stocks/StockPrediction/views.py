from xml.sax.handler import feature_external_ges
from django.shortcuts import render
from StockPrediction.services import machine_algos as ml
# Create your views here.

def index(request):
    return render(request,"index.html")


def getSymbol(request):
    if(request.method == "POST"):
        symbol = request.POST['ticker_name']
        start = request.POST['start']
        end = request.POST['end']
        
        visualize = ml.fetch_data(symbol,start,end)
        moving_average = ml.find7DayMovingAverage(visualize)
        moving_average_100 = ml.find100DayMovingAverage(visualize)
        prediction_l = ml.linearRegression(symbol)
        prediction_ls = ml.stacked_LSTM(symbol)
        context = {
             "name" : symbol,
             "data" :  visualize,
             "start" : start,
             "end" : end,
             "daysavg" : moving_average,
             "hundredaysavg" : moving_average_100,
             "linear_regression_actual" : prediction_l[0],
             "linear_regression_predicted":prediction_l[1],
             "linear_regression_accuracy_score":prediction_l[2],
             "stacked_mean_train" : prediction_ls[2],
             "stacked_mean_test" : prediction_ls[3],
             "stacked_predict_train" : prediction_ls[0],
             "stacked_predict_test"  :prediction_ls[1],
            }
        return render(request,"index.html",context)
    return render(request,"index.html")

