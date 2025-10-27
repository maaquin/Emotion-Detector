# 🧠 Emotion Detector — Facial Emotion Recognition in Real Time

**Reconoce emociones humanas en tiempo real usando IA y visión por computadora.**  
Un proyecto construido con **TensorFlow.js**, **MediaPipe FaceMesh** y **React**, que combina *machine learning* y *front-end interactivo*.

---

## 📖 Descripción

**Emotion Detector** es un proyecto de detección de emociones faciales basado en aprendizaje profundo.  
El modelo fue entrenado con el dataset **FER2013**, procesando imágenes de rostros para clasificar emociones humanas en siete categorías:  

😐 Neutral · 😀 Feliz · 😢 Triste · 😠 Enojado · 😮 Sorprendido · 😨 Miedo · 🤢 Asco  

El flujo principal combina dos componentes:

- **MediaPipe FaceMesh**: detecta 468 puntos (*landmarks*) en el rostro humano y los convierte en datos numéricos.  
- **TensorFlow.js**: recibe esos datos como tensores y predice la emoción correspondiente con un modelo de red neuronal entrenado en Python y exportado a formato `.json` para usarse en la web.

---

## ⚙️ Tecnologías utilizadas

| Área | Tecnologías |
|------|-------------|
| Frontend | React + Vite |
| IA / ML | TensorFlow (Python) · TensorFlow.js |
| Visión por computadora | MediaPipe FaceMesh |
| Preprocesamiento | NumPy · PIL (Python Imaging Library) |
| Backend opcional | Node.js (para despliegues locales o APIs auxiliares) |

---

## 🧩 Cómo funciona

### Entrenamiento del modelo (Python)

- Se usó el dataset **FER2013** con imágenes de 48x48 píxeles en escala de grises.
- El modelo es una red neuronal densa con varias capas y dropout: `Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Dense(128) → Dense(7, softmax)`.
- Se normalizaron los valores X/Y de los 468 landmarks usando StandardScaler antes del entrenamiento.
- Tras entrenar hasta 100 épocas con early stopping y guardado del mejor modelo (best_emotion_model.h5), se exportó con `model.export("emotion_model")` para su uso en TensorFlow.js.
- Durante el entrenamiento, la accuracy de validación osciló entre aproximadamente `0.55–0.56` y la loss entre `1.15–1.17`, reflejando la dificultad de diferenciar algunas emociones similares.

### Predicción en tiempo real (JavaScript)

- El navegador detecta tu rostro con **FaceMesh**.  
- Convierte los *landmarks* en un tensor (`tf.tensor2d`) del tamaño esperado por el modelo.  
- **TensorFlow.js** predice la emoción y la dibuja sobre el video en vivo.

---

## 🚀 Cómo usarlo

### 🧩 1. Clonar el repositorio

```bash
git clone https://github.com/maaquin/EmotionDetector.git
cd EmotionDetector
```

### ⚙️ 2. Instalar dependencias
```bash
npm install
```

### ▶️ 4. Ejecutar el proyecto
```bash
npm run dev
```
- Luego abre tu navegador en http://localhost:5173

## 🌐 Despliegue
- Enlace al proyecto desplegado: [emotion-detector](https://emotion-detector-iota.vercel.app)

---

## ✨ Notas
- **Compatibilidad del navegador:** El proyecto funciona mejor en navegadores modernos que soporten WebGL 2.0 y WebAssembly.
- **Rendimiento:** La predicción en tiempo real puede variar según la cámara y CPU/GPU del dispositivo. Para mejorar la velocidad se puede reducir la resolución del video o usar un modelo más ligero.
- **Precisión del modelo:** El modelo MLP entrenado con FER2013 alcanza una accuracy de test aproximada de `0.55–0.57` y una loss de `1.14–1.17`, lo que significa que puede confundirse entre emociones similares, especialmente con fear, disgust o surprise.
- **Normalización de landmarks:** Los valores X/Y de los 468 puntos faciales fueron normalizados usando StandardScaler, lo que asegura que las coordenadas estén en la misma escala que durante el entrenamiento y mejora la consistencia de las predicciones en tiempo real.
- **Seguridad y privacidad:** Todo el procesamiento ocurre en el navegador; las imágenes de video no se envían a servidores externos.

---

## 📌 Autor
- Luciano Maquin – [@Maaquin](https://github.com/maaquin)