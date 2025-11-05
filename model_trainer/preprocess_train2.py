import os
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Configuración MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
# Lista de generos
IMAGE_ROOT = "imdb/imdb-clean-1024/imdb-clean-1024" 

# Mapeo de etiquetas a números
GENDER_MAP = {'F': 0, 'M': 1}

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

def build_dataset_from_csv(csv_path, image_root_folder):
    X = []
    y = []

    # 1. Cargar el CSV
    df = pd.read_csv(csv_path)

    # 2. Filtrar datos malos
    df = df[df['gender'].isin(['M', 'F'])]
    
    print(f"Procesando {len(df)} caras desde {csv_path}...")

    # 3. Iterar por cada fila (cada cara)
    for index, row in df.iterrows():
        try:
            # 4. Construir la ruta de la imagen
            img_name = row['filename']
            full_path = os.path.join(image_root_folder, img_name)

            # 5. Cargar la imagen COMPLETA
            image = cv2.imread(full_path)
            
            if image is None:
                continue

            # 6. Obtener coordenadas de la cara (asegurar que sean enteros)
            x0 = int(row['x_min'])
            y0 = int(row['y_min'])
            x1 = int(row['x_max'])
            y1 = int(row['y_max'])

            # 7. RECORTAR la cara
            face = image[y0:y1, x0:x1]
            
            # 8. Validar que el recorte no esté vacío
            if face.size == 0:
                continue

            # 9. Pre-procesar la cara
            face_resized = cv2.resize(face, (128, 128)) 
            
            # 10. Obtener la etiqueta (género) y mapearla a 0 o 1
            gender_label = row['gender']
            label = GENDER_MAP[gender_label]

            #12 Extraer los landMarks
            lm = extract_landmarks_from_image_bgr(face_resized)
            if lm is None:
                continue

            # 11. Guardar
            X.append(lm)
            y.append(label)

        except Exception as e:
            file_para_error = row.get('filename', 'FILENAME_DESCONOCIDO')
            print(f"ERROR en {file_para_error}: {e}")

    X = np.array(X)
    y = np.array(y)
    return X, y

# Dataset de entrenamiento
X_train, y_train = build_dataset_from_csv("imdb/imdb_train_new_1024.csv", IMAGE_ROOT)

# Dataset de prueba (evaluación)
X_test, y_test = build_dataset_from_csv("imdb/imdb_test_new_1024.csv", IMAGE_ROOT)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Normalizar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo sencillo
input_dim = X_train.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks útiles
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_gender_model.h5", save_best_only=True)
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
model.export("gender_model")
import joblib
joblib.dump(scaler, "scaler.save")

# Cerrar face_mesh (liberar recursos)
face_mesh.close()


# crear entorno virtual (opcional) python -m venv venv
# activar entorno virtual
# Windows: venv\Scripts\activate
