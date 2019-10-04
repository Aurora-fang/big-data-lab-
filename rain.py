import pandas as pd
import numpy as np

#********** Begin *********#
train=pd.read_csv("./input/train.csv")
test=pd.read_csv("./input/test.csv")
sub=test[["Date","Location"]]
len_train=len(train)
droplist=["Date","Location","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","Cloud9am","Cloud3pm","Rainfall"]

train.drop(droplist,axis=1, inplace=True)
train["RainTomorrow"].loc[train["RainTomorrow"]=="Yes"]=1
train["RainTomorrow"].loc[train["RainTomorrow"] =="No"]=0
y_train=train["RainTomorrow"]
train.drop(["RainTomorrow"],axis=1, inplace=True)
test.drop(droplist,axis=1, inplace=True)

train.fillna(value=0,inplace=True)
test.fillna(value=0,inplace=True)
def clean(df):
    df["Tempjc"]=df["MaxTemp"]-df["MinTemp"]
    df.drop(["MaxTemp","MinTemp"],axis=1, inplace=True)
    df["wdspeed"]=df["WindSpeed3pm"] - df["WindSpeed9am"]
    df.drop(["WindSpeed3pm","WindSpeed9am"],axis=1, inplace=True)
    df["hum"]=df["Humidity3pm"]-df["Humidity9am"]
    df.drop(["Humidity3pm","Humidity9am"],axis=1, inplace=True)
    df["pre"]=df["Pressure3pm"]-df["Pressure9am"]
    df.drop(["Pressure3pm","Pressure9am"],axis=1, inplace=True)
    df["tem"]=df["Temp3pm"]-df["Temp9am"]
    df.drop(["Temp3pm","Temp9am"],axis=1, inplace=True)
    df["RainToday"].loc[df["RainToday"]=="Yes"]=1
    df["RainToday"].loc[df["RainToday"] =="No"]=0
    df["Tempjc"].loc[df["Tempjc"]>=0]=1
    df["Tempjc"].loc[df["Tempjc"] < 0] = 0
    df["wdspeed"].loc[df["wdspeed"]>=0]=1
    df["wdspeed"].loc[df["wdspeed"] < 0] = 0
    df["hum"].loc[df["hum"]>=0]=1
    df["hum"].loc[df["hum"] < 0] = 0
    df["pre"].loc[df["pre"]>=0]=1
    df["pre"].loc[df["pre"]< 0] = 0
    df["tem"].loc[df["tem"]>=0]=1
    df["tem"].loc[df["tem"] < 0] = 0
    return df
train=clean(train)
test=clean(test)
X0=train
y0=y_train
##juzhen yunsuan
X = np.array(X0)
y = np.transpose(np.array([y0]))

Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
pre = np.dot(test, beta)
sub["RainTomorrow"]=pre
##print(sub)
sub.to_csv("./output/test_prediction.csv")
#********** End *********#