const NUM_LANDMARKS = 468;
const INPUT_SHAPE = NUM_LANDMARKS * 2; // 468*2 = 936 entradas

const detector = tf.sequential();

// Capa oculta 1
detector.add(tf.layers.dense({
    units: 128,
    inputShape: [INPUT_SHAPE],
    activation: 'relu'
}));

// Capa oculta 2
detector.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
}));

// Capa de salida (7 emociones)
detector.add(tf.layers.dense({
    units: 7,
    activation: 'softmax'
}));

// Compilar
detector.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

function landmarksToTensor(landmarks) {
    const data = [];
    for (let i = 0; i < landmarks.length; i++) {
        data.push(landmarks[i].x);
        data.push(landmarks[i].y);
    }
    return tf.tensor([data]); // [1, 936]
}

const tensor = landmarksToTensor(faceLandmarks);
const prediction = model.predict(tensor);
const probs = prediction.arraySync()[0]; // [p_neutral, p_happy, ... , p_disgust]