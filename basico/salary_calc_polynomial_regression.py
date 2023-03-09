import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


header1 = ['experience_level', 'salary']
data = [[1, 4500], [2, 5000], [3, 6000], [4, 8000], [5, 11000],[6, 15000],[7, 20000], [8, 30000], [9, 50000], [10, 100000]] 

df = pd.DataFrame(data,columns=header1)
print(df)

# Exemplo Regressão Linear 
fig, ax = plt.subplots(figsize=(5,4))

ax.scatter(df['experience_level'],df['salary'])
ax.set_xlabel('Experience Level')
ax.set_ylabel('Salary')
plt.savefig('1.png', dpi=300)


reg = LinearRegression()
reg.fit(df[['experience_level']],df['salary'])
xekseni=df['experience_level']
yekseni=reg.predict(df[['experience_level']])
ax.plot(xekseni,yekseni,color='green',label='linear regression')
ax.legend()
plt.savefig('2.png',dpi=300)

plt.show()
plt.close()
# Polynomial Regression
polynomial_regression = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regression.fit_transform(df[['experience_level']])

reg = LinearRegression()
reg.fit(x_polynomial,df['salary'])
y_head = reg.predict(x_polynomial)
fig2, ax2 = plt.subplots(figsize=(5,4))
ax2.plot(df['experience_level'], y_head,color='red',label='polynomial regression')
ax2.legend()

ax2.scatter(df['experience_level'],df['salary'])
plt.savefig('degree=12.png',dpi=300)
plt.show()
plt.close()



x_polynomial_new = polynomial_regression.fit_transform([[4.5]])
new_salary = reg.predict(x_polynomial_new)