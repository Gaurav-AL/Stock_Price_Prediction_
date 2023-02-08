from statistics import mode
from types import NoneType
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import os
import math
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
# from StockPrediction.models import Close,PredictedClose

path = os.getcwd()

# print(path)
def fetch_data(ticker_name,start,end):
    try:
        data = yf.download(ticker_name, start=start+'-01-01', end=end +'-01-31')
        data.to_csv(f"{path}\\downloaded_data\\{ticker_name}.csv")
    except ConnectionError as e:
        print("There is Error in Connection")
        
    
    list_data = []
    for row in data['Close']:
        list_data.append(row)
    
    return list_data
# print(fetch_data('GC=F','1980','2022'))

def find7DayMovingAverage(list_data):

    avg = []
    for i in range(7,len(list_data)):
        avg.append(sum(list_data[i-7:i-1]) / 7)
    return avg

def find100DayMovingAverage(list_data):
    avg = []
    for i in range(100,len(list_data)):
        avg.append(sum(list_data[i-100:i-1]) / 100)
    return avg
def linear_regression_train_models(ticker_name):
    url = f"{path}\\media"
    dataset = pd.read_csv(f"{url}/{ticker_name}.csv")
    X,Y = [],[]
    X = dataset[["High","Low","Open","Volume"]]
    Y = dataset["Close"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,shuffle = True)
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    filename = f"{url}/{ticker_name}.sav"
    pickle.dump(clf, open(filename, 'wb'))
# linear_regression_train_models('GC=F')

def linearRegression(ticker_name):
    murl = "F:\stock_project\Stocks\StockPrediction\services\media"
    filename = f"{murl}/{ticker_name}.sav"
    try:
        loaded_model = pickle.load(open(filename, 'rb'))
    except:
        raise("File Not Found Error")
    dataset = pd.read_csv(f"{path}\\downloaded_data\\{ticker_name}.csv")
    X,Y = [],[]
    X = dataset[["High","Low","Open","Volume"]]
    Y = dataset["Close"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.7,shuffle = True)
    prediction = loaded_model.predict(X_test)
    
    dataframe = pd.DataFrame(Y_test,prediction)
    dfr = pd.DataFrame({"Actual":Y_test,"Predicted":prediction})
    # dfr = dict(dfr)
    r2_score = loaded_model.score(X_test,Y_test)
    actual , predicted =  [],  []
    for act in dfr["Actual"]:
        actual.append(act)
    
    for pred in dfr["Predicted"]:
        predicted.append(pred)

    # print(r2_score)
    return [actual,predicted,r2_score*100]

# print(linearRegression('AApL'))
def stacked_LSTM(ticker_name):
    url = f"{path}\\media"
    dataset = pd.read_csv(f"{path}\\downloaded_data\\{ticker_name}.csv")
    # We are using closing price 
    data = dataset.reset_index()["Close"]
    # print(data)
    scaler = MinMaxScaler(feature_range=(0,1))
    
    transformed_data = scaler.fit_transform(np.array(data).reshape(-1,1))
    
    # Splitting dataset
    training_size = int(len(data)*0.65)
    test_size = len(data) -training_size
    
    train_data,test_data = transformed_data[0:training_size,:],transformed_data[training_size:,:1]
    # print(train_data)
    def create_dataset(dataset,timestep=1):
        dataX,dataY = [],[]

        for i in range(len(dataset) - timestep-1):
            temp = dataset[i:(i + timestep) , 0]
            dataX.append(temp)
            dataY.append(dataset[i + timestep ,0])
        return np.array(dataX),np.array(dataY)
    
    timestep = 100
    X_train , y_train = create_dataset(train_data,timestep)
    X_test ,y_test = create_dataset(test_data,timestep)


    
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
    
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True , input_shape = (100,1)))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50))
   
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    # print(model.summary())
    
    model.fit(X_train,y_train, validation_data =(X_test,y_test),epochs=10,verbose = 1,batch_size = 64)
    # model.save_weights(f"{url}\\lstm_model\\lstm_model".format(epoch=0))
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)

    test_predict = scaler.inverse_transform(test_predict)

    mean_train = math.sqrt(mean_squared_error(y_train,train_predict))
    mean_test = math.sqrt(mean_squared_error(y_test,test_predict))
    
    train_predict_l = train_predict.tolist()
    test_predict_l= test_predict.tolist()
    
    predict_train = []
    for l in train_predict_l:
        predict_train.extend(l)
    predict_test = []
    for l in test_predict_l:
        predict_test.extend(l)

    x_input=test_data[int(test_size*0.10):].reshape(1,-1)
    temp_input=list(x_input)
    temp_input = temp_input[0].tolist()
     
    
   
    # n_steps= 100
    
    # # for i in range(2,len(temp_input)):
    # #     if(len(temp_input)%i == 0):
    # #         n_steps = i
    # #         break
    # lst_output=[]
    
    # i=0
    # while(i<30):
    #     # print(x_input)
    #     if(len(temp_input)>50):
    #         # print(temp_input)
    #         x_input = np.array(temp_input[1:])
    #         # print("{} day input {}".format(i,x_input))
    #         x_input=x_input.reshape(1,-1)
    #         x_input = x_input.reshape((1, n_steps, 1))
    #         # print(x_input)
    #         yhat = model.predict(x_input, verbose=0)
    #         # print("{} day output {}".format(i,yhat))
    #         temp_input.extend(yhat[0].tolist())
    #         temp_input=temp_input[1:]
    #         # print(temp_input)
    #         lst_output.extend(yhat.tolist())
    #         i=i+1
    #     else:
    #         x_input = x_input.reshape((1, n_steps,1))
    #         yhat = model.predict(x_input, verbose=0)
    #         # print(yhat[0])
    #         temp_input.extend(yhat[0].tolist())
    #         # print(len(temp_input))
    #         lst_output.extend(yhat.tolist())
    #         i=i+1
        
    # print(predict_train)
    return [predict_train,predict_test,mean_train,mean_test]
    

# stacked_LSTM("MSFT")












   





