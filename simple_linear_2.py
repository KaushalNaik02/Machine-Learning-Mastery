import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

data = {
    'housearea' : [10,20,30,40,50,60,70,80,90,100],
    'price' : [1,2,3,4,5,6,7,8,9,10]
}


df = pd.DataFrame(data)
print(df)



plt.scatter(df['housearea'],df['price'])
plt.xlabel('housearea')
plt.ylabel('price')
plt.show()


x = df[['housearea']]
y = df['price']
print("hello")

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)



print("actual value:",y_test.values)
print("predicted value",y_pred)

mse = mean_squared_error(y_test,y_pred)
print("mean square error is:",mse)

r2 = r2_score(y_test,y_pred)
print("r2 square",r2)


housearea = np.array([[65]])
predicted_score = model.predict(housearea)
print(f"Predicted price for 80 housearea: {predicted_score[0]}")