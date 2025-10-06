# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your stock dataset
data = pd.read_csv("Starbucks Dataset.csv", parse_dates=["Date"], index_col="Date")

# Resample yearly (sum or mean can be used, here we use mean closing price)
resampled_data = data["Close"].resample("Y").mean().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={"Date": "Year", "Close": "ClosePrice"}, inplace=True)

# Extract values
years = resampled_data["Year"].tolist()
prices = resampled_data["ClosePrice"].tolist()

# Prepare variables for linear trend
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, prices)]

n = len(years)
b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]
```
B- POLYNOMIAL TREND ESTIMATION
```
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]

coeff = [[n, sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]

Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)

solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

print(f"Linear Trend: y = {a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add trends to dataframe
resampled_data["Linear Trend"] = linear_trend
resampled_data["Polynomial Trend"] = poly_trend
resampled_data.set_index("Year", inplace=True)

# Plot Linear Trend
resampled_data["ClosePrice"].plot(kind="line", color="blue", marker="o", label="Close Price")
resampled_data["Linear Trend"].plot(kind="line", color="black", linestyle="--", label="Linear Trend")
plt.title("Linear Trend Estimation of Stock Closing Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot Polynomial Trend
resampled_data["ClosePrice"].plot(kind="line", color="blue", marker="o", label="Close Price")
resampled_data["Polynomial Trend"].plot(kind="line", color="red", marker="o", label="Polynomial Trend")
plt.title("Polynomial Trend Estimation (Degree 2) of Stock Closing Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="1113" height="834" alt="Screenshot 2025-08-30 092004" src="https://github.com/user-attachments/assets/73c89b19-a966-437e-a85f-0b12741fd067" />

B- POLYNOMIAL TREND ESTIMATION
<img width="1127" height="832" alt="Screenshot 2025-08-30 092033" src="https://github.com/user-attachments/assets/b4297ccd-fbe2-4cf5-8684-eb2681d61176" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
