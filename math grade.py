import pandas as pd
import numpy as np
train=pd.read_csv("train.csv")
test=pd.read_csv('test.csv')


x_train=train[['reading score','writing score']]
x_test=test[['reading score','writing score']]
y_train=train['math score']
x_train['average']=(x_train['reading score']+x_train['writing score'])/2
x=x_train['average']
x_test['average']=(x_test['reading score']+x_test['writing score'])/2
x_test=x_test['average']
poly = np.polyfit(x,y_train,deg=2)
z = np.polyval(poly,x_test)
result=pd.DataFrame({'id':test['id'],'math score':z})
##LRresult.to_csv('LR_pre.csv', index=False)






