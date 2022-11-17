import csv 
import pandas as pd 
from sklearn import linear_model
import numpy as np 

header1 = ['area', 'room_count', 'building_age', 'price']
data = [[180, 5, 10, 510000], [225, 4, 18, 508000], [260, 3, 2, 548000], [320, 6, 10, 650000], [335, 4, 9, 628000],[345, 5, 14, 632000],[355, 3, 0, 680000], [400, 4, 10, 725000]] 

df = pd.DataFrame(data,columns=header1)
print(df)

# Selecionar features 
# No caso é a área, quantidade de quartos, idade da casa

# O valor que será previsto é o preço
features1 = df[['area', 'room_count', 'building_age']]
value2predict = df['price']

reg = linear_model.LinearRegression()

reg.fit(features1, value2predict)

data2 = [[230,4,10],[250,2,15]]
df2 = pd.DataFrame(data2,columns=header1[0:-1])

casa_nova1 = reg.predict([df2.values[0],df2.values[1]])

casa_nova2 = reg.predict([[250,8,0]])


# Agora veremos os coeficientes para a formula da regressão multilinear 

reg.coef_


a = reg.intercept_

b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 250
x2 = 8
x3 = 0

y = a + b1*x1 + b2*x2 + b3*x3 
