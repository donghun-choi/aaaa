import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
import PIL
import cv2
import PIL.Image

abalone_model = tf.keras.Sequential([
  layers.Flatten(input_shape=(224, 224, 1)),
  layers.Dense(224),
  layers.Dense(224),
  layers.Dense(224),
  layers.Dense(128),
  layers.Dense(64),
  layers.Dense(4)
])
abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),optimizer = tf.keras.optimizers.Adam())

def load_img(index):
    # 이미지 경로 설정
    image_path = './frames/{}.jpg'.format(index)
    
    # 이미지 불러오기
    img = cv2.imread(image_path)
    
    # 이미지 크기 조정
    img = cv2.resize(img, (224, 224))
    
    # Canny Edge Detection 적용
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    
    return edges

def load_wasd(index):
    file_path = './output.csv'
    
    file = pd.read_csv(file_path)
    row = file[file["frame_count"] == index]
    
    wasd_values = row.iloc[0, 1:].tolist()  # Assuming WASD columns start from index 2
    return wasd_values

def get_val(index):
    x_data = load_img(index)
    y_data = load_wasd(index)
    return x_data,y_data

# Load the CSV file
df = pd.read_csv('./output.csv')

# Get all the frame_count values
frame_counts = df['frame_count'].values

# Create lists to store the images and labels
x_data = []
y_data = []

# Load all the images and labels
for index in frame_counts:
    print(index)
    x, y = get_val(index)
    x_data.append(x)
    y_data.append(y)

# Convert lists to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Reshape the x_data to fit the model input shape
x_data = x_data.reshape(-1, 224, 224, 1)

# Train the model
abalone_model.fit(x_data, y_data, epochs=10)

# Save the model
abalone_model.save('abalone_model.h5')

from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model('abalone_model_v2.h5')

# Now you can use the loaded model to predict the WASD values for a new image
def predict_wasd(img):
    # Preprocess the image the same way we did with the training images
    img = cv2.resize(img, (224, 224))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edges = edges.reshape(-1, 224, 224, 1)
    
    # Use the model to predict the WASD values
    wasd_values = loaded_model.predict(edges)
    
    return wasd_values
