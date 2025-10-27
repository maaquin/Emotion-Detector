import os
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
# Lista de emociones (asegúrate de que tus carpetas coincidan)
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']

def extract_landmarks_from_image_bgr(image_bgr):
    """
    image_bgr: imagen cargada por cv2 (BGR)
    Retorna: vector 468*2 (x,y) normalizados o None si no se detecta rostro
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0].landmark
    # Extraer x,y en formato plano
    lm = []
    for p in face_landmarks:
        lm.extend([p.x, p.y])
    return np.array(lm, dtype=np.float32)

def build_dataset_from_folder(root_folder):
    X = []
    y = []
    for label_idx, emotion in enumerate(EMOTIONS):
        folder = os.path.join(root_folder, emotion)
        if not os.path.isdir(folder):
            print(f"Advertencia: no existe la carpeta {folder}, se salta.")
            continue
        for fname in os.listdir(folder):
            if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
                continue
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                print("No se pudo leer:", fpath)
                continue
            lm = extract_landmarks_from_image_bgr(img)
            if lm is None:
                continue
            X.append(lm)
            y.append(label_idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Dataset de entrenamiento
X_train, y_train = build_dataset_from_folder("fer2013/train")

# Dataset de prueba (evaluación)
X_test, y_test = build_dataset_from_folder("fer2013/test")

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Normalizar (muy importante para redes densas)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo sencillo (MLP)
input_dim = X_train.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks útiles
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_emotion_model.h5", save_best_only=True)
]

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks)

# Evaluar modelo
loss, acc = model.evaluate(X_test, y_test)
print("Test loss, acc:", loss, acc)


# Guardar modelo
model.export("emotion_model")
import joblib
joblib.dump(scaler, "scaler.save")

# Cerrar face_mesh (liberar recursos)
face_mesh.close()