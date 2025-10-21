import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Función para cargar imágenes
def load_images(folder_path):
    X = []
    y = []
    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    for idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(folder_path, emotion)
        for file in os.listdir(emotion_path):
            if file.endswith(".png") or file.endswith(".jpg"):
                img = Image.open(os.path.join(emotion_path, file)).convert("L")
                img = img.resize((48,48))
                X.append(np.array(img)/255.0)
                y.append(idx)
    X = np.array(X).reshape(-1, 48*48)
    y = np.array(y)
    return X, y

X_train, y_train = load_images("fer2013/train")
X_test, y_test = load_images("fer2013/test")

# Definir modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(48*48,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Guardar modelo
model.export("emotion_model")
