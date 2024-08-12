import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

data = {
    'SquareFootage': [1500,1600, 1700, 1800,1900,2000, 2100,2200,2300, 2400],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

df = pd.DataFrame(data)
print(df)


plt.scatter(df['SquareFootage'],df['Price'],color ='blue')
plt.xlabel('SquareFootage')
plt.ylabel('Price')
plt.show()


x = df[['SquareFootage']]
y = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Actual VAlues", y_test.values)
print("Predicted Values", y_pred)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

SquareFootage = np.array([[3000]])
predicted_score = model.predict(SquareFootage)
print(f"Predicted price for 3000 SquareFootage: {predicted_score[0]}")