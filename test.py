import cv2
import numpy as np
import tensorflow as tf
import pickle


model = tf.keras.models.load_model("food_model.h5")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)


test_image_path = "D:\\hehe.jpg"  
img = cv2.imread(test_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, 0)


pred = model.predict(img)
food = labels[np.argmax(pred)]
print("Predicted Food:", food)