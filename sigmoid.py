############ IMPORTS ############

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


############ MAIN CODE ############

# Estimated data points from the plot (replace with data for each participant)
x_data = np.array([0.063, 0.0912, 0.132, 0.191, 0.2764, 0.4])
y_data = np.array([0.2, 0.55, 0.6, 0.9, 1.0, 1.0])

# Sigmoid function definition
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))

# Fit the sigmoid function to the data
params, _ = curve_fit(sigmoid, x_data, y_data)
a, b = params

# Generate x values for the fitted curve
x_fit = np.linspace(0, 0.5, 100)
y_fit = sigmoid(x_fit, a, b)

# Compute the accuracy of the fitting
y_pred = sigmoid(x_data, a, b)
accuracy = np.mean((y_pred - y_data) ** 2)

# Calculate the inflection point
inflection_point = -b / a
inflection_value = sigmoid(inflection_point, a, b)

# Plot the original data and the fitted sigmoid function
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', s=100, zorder=5)
plt.plot(x_fit, y_fit, color='red', label='Fitted Sigmoid', zorder=1)
plt.axvline(x=inflection_point, color='green', linestyle='--', label='Threshold', zorder=2)
plt.title('Temporal Modulations Psychometric Curve - Participant 3')
plt.xlabel('Modulation Level')
plt.ylabel('Proportion Correct')
plt.legend()
plt.show()

print(f'Fitted Sigmoid: $a={a:.2f}, b={b:.2f}$')
print(f"Best Sigmoid Fit (MSE: {accuracy:.4f})")
print(f"Inflection Point: x={inflection_point:.4f}, y={inflection_value:.4f}")