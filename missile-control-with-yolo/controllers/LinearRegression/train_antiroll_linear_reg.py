# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 09:04:26 2024

@author: ammar
"""

# In[data preprocessing]

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

columns_to_extract = ['roll', 'roll rate', 'aileron input']
filepath = 'black_box/combined_output.xlsx'
training_data = pd.read_excel(filepath, usecols=columns_to_extract)


# MAKE SURE THE LABEL DATA IS IN THE LAST COLUMN!
X = training_data.iloc[:, :-1]  # Select all columns except the last one as input (2 columns)
y = training_data.iloc[:, -1]   # Select the last column as the label (1 column)

# Split the data into training and validation sets (optional but recommended)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)


# In[linear regression]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)


# In[Evaluate the model]
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)  # Mean Squared Error
r2 = r2_score(y_val, y_pred)             # R-squared Score

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')



# In[visualize 1]

import matplotlib.pyplot as plt
import numpy as np

# Step 1: Plot the actual vs predicted values
plt.figure(figsize=(10, 6))

# Plot actual values (in red)
plt.plot(np.arange(len(y_val)), y_val, label='Actual Values', color='red')

# Plot predicted values (in blue)
plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted Values', color='blue')

# Add labels and title
plt.xlabel('Data Points')
plt.ylabel('Aileron Input (Target)')
plt.title('Actual vs Predicted Values - Linear Regression')

# Add a legend
plt.legend()

# Display the plot
plt.show()

# In[visualize 2]

import matplotlib.pyplot as plt
import numpy as np

# Extract the first feature (roll) for visualization
X_train_roll = X_train[:, 0]  # First column (roll) of the training data
X_val_roll = X_val[:, 0]      # First column (roll) of the validation data

# Step 1: Scatter plot of actual data (roll vs aileron input)
plt.figure(figsize=(10, 6))
plt.scatter(X_val_roll, y_val, color='red', label='Actual Values')

# Step 2: Plot the predicted regression line (use the model to predict based on the first feature)
y_pred_line = model.predict(X_val)

plt.plot(X_val_roll, y_pred_line, color='blue', label='Predicted Line')

# Step 3: Add labels and title
plt.xlabel('Roll (First Input Feature)')
plt.ylabel('Aileron Input (Target)')
plt.title('Roll vs Aileron Input with Predicted Regression Line')

# Add a legend
plt.legend()

# Display the plot
plt.show()


# In[save the model]
import joblib

# Save the model to a file
joblib.dump(model, 'antiroll_models/linear_regression_model.pkl')





