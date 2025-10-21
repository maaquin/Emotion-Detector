import { useEffect, useRef, useState } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import "./styles.css";

export const Dashboard = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const cameraRef = useRef(null);
    const faceMeshRef = useRef(null);

    const [isCameraOn, setIsCameraOn] = useState(false);
    const [isFaceMeshOn, setIsFaceMeshOn] = useState(false);

    const [model, setModel] = useState(null);

    useEffect(() => {
        // Carga el modelo al montar el componente
        const loadModel = async () => {
            const m = await tf.loadGraphModel("../../tfjs_model/model.json");
            setModel(m);
        };
        loadModel();
    }, []);


    useEffect(() => {
        // Inicializa FaceMesh una sola vez
        const faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
            },
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        faceMesh.onResults(onResults);
        faceMeshRef.current = faceMesh;

        function onResults(results) {
            const canvasCtx = canvasRef.current.getContext("2d");
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            // Efecto espejo
            // canvasCtx.scale(-1, 1);
            // canvasCtx.translate(-canvasRef.current.width, 0);

            // Dibuja el video base
            canvasCtx.drawImage(
                results.image,
                0,
                0,
                canvasRef.current.width,
                canvasRef.current.height
            );

            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                const landmarks = results.multiFaceLandmarks[0];

                // mínimo y máximo en X e Y
                let minX = Infinity, minY = Infinity;
                let maxX = -Infinity, maxY = -Infinity;

                for (const point of landmarks) {
                    const x = point.x * canvasRef.current.width;
                    const y = point.y * canvasRef.current.height;

                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }

                //rectángulo que rodea la cara
                const canvasCtx = canvasRef.current.getContext("2d");
                const width = maxX - minX;
                const height = maxY - minY;

                canvasCtx.strokeStyle = "rgba(201, 7, 36, 0.88)";
                canvasCtx.lineWidth = 3;
                canvasCtx.strokeRect(minX, minY, width, height);

                if (model) {
                    const tensor = landmarksToImageTensor(landmarks);
                    const prediction = model.predict(tensor);
                    const probs = prediction.arraySync()[0];

                    // Obtener la emoción con mayor probabilidad
                    const emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust'];
                    const maxIndex = probs.indexOf(Math.max(...probs));
                    const emotion = emotions[maxIndex];

                    // Dibujar la emoción en la cara
                    canvasCtx.font = "20px Arial";
                    canvasCtx.fillStyle = "red";
                    canvasCtx.fillText(emotion, minX, minY - 10);

                    tensor.dispose();
                    prediction.dispose();
                }
            }

            canvasCtx.restore();
        }

        return () => {
            // Limpieza al desmontar
            if (cameraRef.current) cameraRef.current.stop();
        };
    }, [isFaceMeshOn, model]);

    function landmarksToTensor(landmarks) {
        const data = [];
        let flat = landmarks.flatMap(p => [p.x, p.y, p.z]);
        while (flat.length < 2304) flat.push(0);
        for (let i = 0; i < landmarks.length; i++) {
            data.push(landmarks[i].x);
            data.push(landmarks[i].y);
        }
        return tf.tensor([data]); // [1, 936]
    }

    function landmarksToImageTensor(landmarks) {
        // Crear un canvas virtual de 48x48
        const canvas = document.createElement("canvas");
        canvas.width = 48;
        canvas.height = 48;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, 48, 48);

        // Dibujar landmarks
        ctx.fillStyle = "white";
        for (const point of landmarks) {
            const x = Math.floor(point.x * 48);
            const y = Math.floor(point.y * 48);
            ctx.fillRect(x, y, 1, 1);
        }

        // Obtener datos de imagen
        const imgData = ctx.getImageData(0, 0, 48, 48);
        const pixels = Array.from(imgData.data)
            .filter((_, i) => i % 4 === 0) // Tomar solo canal R
            .map(v => v / 255); // Normalizar

        return tf.tensor2d([pixels]);
    }

    // 🔘 Alternar cámara
    const toggleCamera = async () => {
        if (!isCameraOn) {
            const camera = new Camera(videoRef.current, {
                onFrame: async () => {
                    if (faceMeshRef.current) {
                        await faceMeshRef.current.send({ image: videoRef.current });
                    }
                },
                width: 640,
                height: 480,
            });
            cameraRef.current = camera;
            await camera.start();
            setIsCameraOn(true);
        } else {
            cameraRef.current?.stop();
            setIsCameraOn(false);
        }
    };

    const toggleFaceMesh = () => {
        setIsFaceMeshOn((prev) => !prev);
    };

    return (
        <div className="dashboard-container">
            <h1>Detección de emociones</h1>
            <div className="video">

                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="rounded-2xl shadow-lg w-[640px] h-[480px] transform -scale-x-100"
                />
                <canvas
                    ref={canvasRef}
                    width="640"
                    height="480"
                    className="absolute top-0 left-0 rounded-2xl"
                />

            </div>
            <div>
                <button onClick={toggleCamera}>
                    camera
                </button>
            </div>
        </div>
    );
};