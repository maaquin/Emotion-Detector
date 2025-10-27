import { useEffect, useRef, useState } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";

import no_camera from "../assets/no_video.png";
import scalerParams from '../../public/tfjs_model/scaler.json';
import "./styles.css";

export const Dashboard = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const cameraRef = useRef(null);
    const faceMeshRef = useRef(null);

    const [isCameraOn, setIsCameraOn] = useState(false);
    const [isFaceMeshOn, setIsFaceMeshOn] = useState(false);
    const [isExpand, setIsExpand] = useState(false);

    const [model, setModel] = useState(null);

    useEffect(() => {
        // Carga el modelo al montar el componente
        const loadModel = async () => {
            const m = await tf.loadGraphModel("../../public/tfjs_model/model.json");
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

                canvasCtx.strokeStyle = "#fff";
                canvasCtx.lineWidth = 3;
                canvasCtx.strokeRect(minX, minY, width, height);

                if (model) {
                    const tensor = landmarksToTensor(landmarks);
                    const prediction = model.predict(tensor);
                    const probs = prediction.arraySync()[0];

                    const emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust'];

                    // Obtener las dos emociones con mayores probabilidades
                    const maxIndex = probs.indexOf(Math.max(...probs));
                    const maxEmotion = emotions[maxIndex];
                    const maxPercent = Math.round(probs[maxIndex] * 100);

                    const second = Math.max(...probs.filter(v => v !== probs[maxIndex]));
                    const secondIndex = probs.indexOf(second);
                    const secondEmotion = emotions[secondIndex];
                    const secondPercent = Math.round(probs[secondIndex] * 100);

                    // Dibujar la emoción en la cara 
                    canvasCtx.font = "20px Arial";
                    canvasCtx.fillStyle = "#fff";
                    canvasCtx.fillText(`${maxEmotion}: ${maxPercent}%`, minX, minY - 10);
                    canvasCtx.fillText(`${secondEmotion}: ${secondPercent}%`, minX, minY - 10 - 18);

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
        const data = landmarks.flatMap(p => [p.x, p.y]);
        const normalized = data.map((v, i) => (v - scalerParams.mean[i]) / scalerParams.scale[i]);
        return tf.tensor([normalized]); // [1, 936]
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

    return (
        <div className="dashboard-container">
            <div className={isExpand ? 'expanded-window' : 'window'}>
                <div className="header">
                    <h1>Detección de emociones</h1>
                    <div className="btns">
                        <button className="btn" onClick={() => setIsExpand(prev => !prev)}>
                            <div className="expand"/>
                        </button>
                        <button className="btn" onClick={() => window.location.href = 'https://www.google.com'}>
                            <div className="close"/>
                        </button>
                    </div>
                </div>
                <div className='video'>
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

                    {!isCameraOn &&
                        <div className="no_camera">
                            <span>Cámara apagada</span>
                            <img src={no_camera} alt="no camera img" />
                        </div>
                    }
                    
                    <button className="camara" onClick={toggleCamera}>
                        {isCameraOn ? 'apagar' : 'encender'}
                    </button>
                </div>
            </div>
        </div>
    );
};