# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:55:56 2024

@author: ammar
"""

# In[data preprocessing]

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

timesteps = 10  # Number of time steps (sequence length)
features = 2    # Number of features per timestep

columns_to_extract = ['roll', 'roll rate', 'aileron input']

filepath = 'black_box/combined_output.xlsx'
training_data = pd.read_excel(filepath, usecols=columns_to_extract)
# X = training_data.iloc[10, -1]

X_train = []
y_train = []
for i in range(len(training_data)-timesteps):
    X_train.append(training_data.iloc[i:i+timesteps, :2].to_numpy())
    y_train.append(training_data.iloc[i+timesteps-1, -1])
y_train = np.asarray(y_train)


X_train = np.reshape(X_train, (len(X_train), timesteps, features))
y_train = np.reshape(y_train, (len(y_train), 1))

# In[build model]

# Define the model
inputs = Input(shape=(timesteps, features))

# Add RNN layers
x = layers.BatchNormalization()(inputs)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.LSTM(64)(x)
x = layers.BatchNormalization()(x)
# x = layers.LSTM(16)(x)
# x = layers.BatchNormalization()(x)

# # Add dense layers
# x = layers.Dense(32, activation='relu')(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(32, activation='relu')(x)
# x = layers.BatchNormalization()(x)

# Output layer
outputs = layers.Dense(1, activation='linear')(x)

# Create the model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss='mean_squared_error')

model.summary()

# In[train the model]
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
checkpoint_callback = ModelCheckpoint(
    # Filepath where the model will be saved
    filepath=f'antiroll_models/best_antiroll_model_lstm_{features}_inputs_{timesteps}t_{timestr}.h5',
    monitor='val_loss',                         # Metric to monitor (you can change this)
    save_best_only=True,                        # Only save the model when 'val_loss' improves
    save_weights_only=False,                    # Set this to True if you only want to save weights
    mode='min',                                 # 'min' for 'val_loss', 'max' for 'val_accuracy'
    verbose=1                                   # Print a message when saving the best model
)

# ReduceLROnPlateau callback
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',      # Metric to monitor
    factor=0.5,              # Factor by which the learning rate will be reduced (lr = lr * factor)
    patience=16,             # Number of epochs with no improvement after which the learning rate will be reduced
    min_lr=1e-8,             # Lower bound on the learning rate
    verbose=1                # Print a message when reducing the learning rate
)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_split=0.1, 
                    callbacks=[checkpoint_callback, reduce_lr_callback]
                    )


# In[1] display the history

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# In[2] save the model
model.save(f'antiroll_models/antiroll_model_lstm_{features}_inputs_{timesteps}t_{timestr}.h5')




