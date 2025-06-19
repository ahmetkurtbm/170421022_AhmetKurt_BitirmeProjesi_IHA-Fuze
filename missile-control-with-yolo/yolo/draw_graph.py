# -*- coding: utf-8 -*-
"""
Created on Sun May 25 23:24:04 2025

@author: ammar
"""

import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Read the CSV file ===
csv_path = "D:/Projects/missile-control-with-yolo/yolo/YOLOv8_M/results.csv"
df = pd.read_csv(csv_path)

# === Step 2: Extract relevant columns ===
epochs = df['                  epoch'] if '                  epoch' in df.columns else df['epoch']  # sometimes spaces are there

train_box_loss = df['         train/box_loss']
val_box_loss = df['           val/box_loss']

train_dfl = df['         train/dfl_loss']
val_dfl = df['           val/dfl_loss']

precision = df['   metrics/precision(B)']
recall = df['      metrics/recall(B)']
map50 = df['       metrics/mAP50(B)']
map5095 = df['    metrics/mAP50-95(B)']

# === Step 3: Plotting ===
plt.figure(figsize=(16, 8))

# Plot 1: Train & Validation box loss
plt.subplot(2, 3, 1)
plt.plot(epochs, train_box_loss, label='Train Box Loss', color='blue')
plt.plot(epochs, val_box_loss, label='Val Box Loss', color='orange')
plt.title('Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Precision
plt.subplot(2, 3, 2)
plt.plot(epochs, precision, label='Precision (B)', color='blue')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.grid(True)

# Plot 3: Recall
plt.subplot(2, 3, 3)
plt.plot(epochs, recall, label='Recall (B)', color='blue')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.grid(True)

# Plot 4: Dfl loss
plt.subplot(2, 3, 4)
plt.plot(epochs, train_dfl, label='Train DFL Loss', color='blue')
plt.plot(epochs, val_dfl, label='Val DFL Loss', color='orange')
plt.title('DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 5: mAP@0.5
plt.subplot(2, 3, 5)
plt.plot(epochs, map50, label='mAP@0.5 (B)', color='blue')
plt.title('mAP@0.5')
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.grid(True)

# Plot 6: mAP@0.5:0.95
plt.subplot(2, 3, 6)
plt.plot(epochs, map5095, label='mAP@0.5:0.95 (B)', color='blue')
plt.title('mAP@0.5:0.95')
plt.xlabel('Epoch')
plt.ylabel('mAP50-95')
plt.grid(True)

# Final layout adjustments
plt.tight_layout()
plt.suptitle("YOLOv8m Training Metrics", fontsize=16, y=1.02)
plt.show()
