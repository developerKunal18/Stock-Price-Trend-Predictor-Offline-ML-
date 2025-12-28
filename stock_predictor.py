import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample stock price data
data = {
    "day": [1,2,3,4,5,6,7,8,9,10],
    "price": [100,102,101,105,108,110,109,112,115,117]
}

df = pd.DataFrame(data)

X = df[["day"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

tomorrow = [[11]]
prediction = model.predict(tomorrow)

print("💹 Stock Trend Predictor — Day 75\n")
print(f"Predicted price for tomorrow: ₹{prediction[0]:.2f}")

trend = "UP 📈" if prediction[0] > df["price"].iloc[-1] else "DOWN 📉"
print("Expected trend:", trend)
