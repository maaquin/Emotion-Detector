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
- El modelo es una red neuronal densa: `Dense(128) → Dense(64) → Dense(7, softmax)`  
- Tras el entrenamiento, se exportó con `model.export("emotion_model")` para su uso en TensorFlow.js.

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
- Enlace al proyecto desplegado: [Por agregar]

---

## ✨ Notas
- **Compatibilidad del navegador:** El proyecto funciona mejor en navegadores modernos que soporten WebGL 2.0 y WebAssembly.
- **Rendimiento:** La predicción en tiempo real puede variar según la cámara y CPU/GPU del dispositivo. Para mejorar la velocidad se puede reducir la resolución del video o usar un modelo más ligero.
- **Precisión del modelo:** El modelo actual es un MLP (red densa) entrenado con FER2013. Puede confundirse con emociones similares; usar CNNs mejoraría el reconocimiento.
- **Seguridad y privacidad:** Todo el procesamiento ocurre en el navegador; las imágenes de video no se envían a servidores externos.
- **Extensiones futuras:** Se puede integrar con sistemas de análisis de emociones, chatbots o apps de psicología/educación.
- **Errores comunes:** Si model.predict lanza un error de shape, revisa que los landmarks se conviertan correctamente en un tensor de tamaño [1, 2304].

---

## 📌 Autor
- Luciano Maquin – [@Maaquin](https://github.com/maaquin)