import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pickle


dataset_path = "D:\\archive\\Indian Food Images\\Indian Food Images"  
image_size = 224
batch_size = 32
epochs = 20


selected_foods = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print(f"Found {len(selected_foods)} food categories: {selected_foods}")


X = []
y = []

print("Loading images...")
for idx, food in enumerate(selected_foods):
    folder_path = os.path.join(dataset_path, food)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            img = img / 255.0  
            X.append(img)
            y.append(idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(selected_foods))

print(f"Loaded {len(X)} images.")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(selected_foods), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)


for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)


model.save("food_model.h5")
print("Model saved as food_model.h5")


with open("labels.pkl", "wb") as f:
    pickle.dump(selected_foods, f)
print("Labels saved as labels.pkl")