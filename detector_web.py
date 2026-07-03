from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB máximo

# Cargar modelo YOLO oficial (Nano, rápido y liviano)
modelo = YOLO("yolov8n.pt")

CLASES_VEHICULOS = {
    2: {"name": "Auto", "color": (0, 255, 0)},       # Verde
    3: {"name": "Motocicleta", "color": (255, 0, 0)}, # Azul
    5: {"name": "Autobús", "color": (0, 165, 255)},   # Naranja
    7: {"name": "Camión", "color": (0, 0, 255)}       # Rojo
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 Detector Vehicular Pro YOLO</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            padding: 30px;
            max-width: 1100px;
            width: 100%;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        .main-panel { grid-column: span 1; }
        .side-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #e9ecef;
            max-height: 70vh;
            overflow-y: auto;
        }
        @media (max-width: 850px) {
            .container { grid-template-columns: 1fr; }
            .side-panel { max-height: none; }
        }
        h1 { color: #1e3c72; margin-bottom: 5px; font-size: 2.2em; text-align: center; }
        .subtitle { text-align: center; color: #666; margin-bottom: 25px; font-size: 1em; }

        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #eee; }
        .tab-btn {
            padding: 12px 20px; background: none; border: none; cursor: pointer;
            font-size: 1em; color: #999; font-weight: bold; border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        .tab-btn.active { color: #2a5298; border-bottom-color: #2a5298; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Controles extras */
        .controls-panel {
            background: #f1f3f9; padding: 15px; border-radius: 10px; margin-bottom: 20px;
            display: flex; align-items: center; justify-content: space-between; gap: 15px;
        }
        .slider-group { display: flex; align-items: center; gap: 10px; width: 100%; }
        .slider-group input { flex-grow: 1; }

        .upload-area {
            border: 3px dashed #2a5298; border-radius: 10px; padding: 40px;
            text-align: center; cursor: pointer; transition: all 0.3s ease; background: #fdfdfd;
        }
        .upload-area:hover, .upload-area.dragover { border-color: #1e3c72; background: #f0f4f8; transform: scale(1.01); }
        input[type="file"] { display: none; }

        .image-preview { width: 100%; border-radius: 10px; margin-top: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }

        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }
        .stat-box { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-number { font-size: 1.8em; font-weight: bold; }

        .history-item {
            background: white; border-radius: 8px; padding: 10px; margin-bottom: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 5px solid #2a5298;
            font-size: 0.9em; display: flex; justify-content: space-between; align-items: center;
        }

        .btn {
            background: #2a5298; color: white; border: none; padding: 10px 20px;
            border-radius: 6px; cursor: pointer; font-weight: bold; transition: background 0.2s;
        }
        .btn:hover { background: #1e3c72; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #218838; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }

        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #2a5298; border-radius: 50%;
            width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 10px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #dc3545; color: white; padding: 12px; border-radius: 8px; margin-top: 15px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-panel">
            <h1>🚗 Detector Vehicular Pro</h1>
            <p class="subtitle">Análisis inteligente de tráfico con YOLOv8</p>

            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab(event, 'image')">📸 Procesar Imagen</button>
                <button class="tab-btn" onclick="switchTab(event, 'webcam')">🎥 Cámara en Vivo</button>
            </div>

            <div class="controls-panel">
                <div class="slider-group">
                    <label for="confRange">🎯 Confianza Mínima: <span id="confVal">50</span>%</label>
                    <input type="range" id="confRange" min="25" max="90" value="50" oninput="document.getElementById('confVal').textContent=this.value">
                </div>
                <button id="downloadBtn" class="btn btn-success" style="display:none;" onclick="downloadJsonReport()">📊 Exportar JSON</button>
            </div>

            <div id="image" class="tab-content active">
                <div class="upload-area" onclick="document.getElementById('imageInput').click();"
                     ondrop="handleDrop(event)" ondragover="event.preventDefault()" ondragleave="event.preventDefault()">
                    <div style="font-size: 2.5em;">📤</div>
                    <div style="font-weight:bold; color:#2a5298; margin: 8px 0;">Arrastra o selecciona un archivo</div>
                    <div style="font-size:0.85em; color:#888;">Formatos válidos: JPG, JPEG, PNG</div>
                    <input type="file" id="imageInput" accept="image/*" onchange="handleImageSelect(event)">
                </div>
                <div id="loading" style="display:none; text-align:center; margin-top:20px;">
                    <div class="spinner"></div>
                    <p>Procesando con Inteligencia Artificial...</p>
                </div>
                <div id="error" class="error"></div>
                <img id="resultImage" class="image-preview" style="display:none;">
            </div>

            <div id="webcam" class="tab-content">
                <div style="text-align: center;">
                    <video id="webcamVideo" width="100%" style="display:none; border-radius:10px;" autoplay muted playsinline></video>
                    <canvas id="webcamCanvas" width="640" height="480" style="width:100%; border-radius:10px; display:none;"></canvas>
                    <div id="webcamControls" style="margin-top:15px;">
                        <button id="startWebcamBtn" class="btn" onclick="startWebcam()">Iniciar Cámara</button>
                        <button id="stopWebcamBtn" class="btn btn-danger" onclick="stopWebcam()" style="display:none;">Detener Cámara</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="side-panel">
            <h3 style="color:#1e3c72; margin-bottom:15px; border-bottom: 2px solid #2a5298; padding-bottom:5px;">📊 Panel de Control</h3>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number" id="totalCount">0</div>
                    <div style="font-size:0.8em; opacity:0.9;">Detectados</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="avgConfidence">0%</div>
                    <div style="font-size:0.8em; opacity:0.9;">Confianza Prom.</div>
                </div>
            </div>

            <h4 style="margin: 15px 0 10px 0; color:#555;">📋 Últimos Eventos:</h4>
            <div id="historyLog">
                <p style="color:#aaa; font-style:italic; font-size:0.9em; text-align:center; margin-top:20px;">No hay registros aún</p>
            </div>
        </div>
    </div>

    <script>
        let currentStream = null;
        let webcamInterval = null;
        let lastApiResponseData = null; // Guardará el último resultado para exportar JSON

        function switchTab(e, tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            e.currentTarget.classList.add('active');
            if(tabName !== 'webcam') stopWebcam();
        }

        function handleDrop(e) {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                document.getElementById('imageInput').files = e.dataTransfer.files;
                sendImageToServer(e.dataTransfer.files[0]);
            }
        }

        function handleImageSelect(e) {
            if(e.target.files.length > 0) sendImageToServer(e.target.files[0]);
        }

        function sendImageToServer(file) {
            const loading = document.getElementById('loading');
            const errorDiv = document.getElementById('error');
            const resultImg = document.getElementById('resultImage');
            const conf = document.getElementById('confRange').value;

            loading.style.display = 'block';
            errorDiv.style.display = 'none';
            resultImg.style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);
            formData.append('conf', conf / 100);

            fetch('/detect', { method: 'POST', body: formData })
            .then(r => r.json())
            .then(data => {
                loading.style.display = 'none';
                if(data.error) { showErrorMessage(data.error); return; }

                lastApiResponseData = data;
                resultImg.src = 'data:image/jpeg;base64,' + data.image;
                resultImg.style.display = 'block';
                document.getElementById('downloadBtn').style.display = 'inline-block';

                updateMetricsAndHistory(data);
            })
            .catch(err => {
                loading.style.display = 'none';
                showErrorMessage('Error de conexión con el servidor.');
            });
        }

        function updateMetricsAndHistory(data) {
            document.getElementById('totalCount').textContent = data.total_detections;
            document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(1) + '%';

            const log = document.getElementById('historyLog');
            if(data.detections.length === 0) return;

            if(log.innerHTML.includes('No hay registros')) log.innerHTML = '';

            data.detections.forEach(det => {
                const timeStr = new Date().toLocaleTimeString();
                log.insertAdjacentHTML('afterbegin', `
                    <div class="history-item">
                        <div><strong>${det.class}</strong> <span style="color:#666; font-size:0.8em;">[${timeStr}]</span></div>
                        <div style="color:#2a5298; font-weight:bold;">${det.confidence.toFixed(1)}%</div>
                    </div>
                `);
            });
        }

        function showErrorMessage(msg) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = '❌ ' + msg;
            errorDiv.style.display = 'block';
        }

        // CONTROL WEBCAM OPTIMIZADO (Anti-Saturación)
        function startWebcam() {
            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then(stream => {
                currentStream = stream;
                const video = document.getElementById('webcamVideo');
                video.srcObject = stream;
                video.style.display = 'block';
                document.getElementById('webcamCanvas').style.display = 'block';
                document.getElementById('startWebcamBtn').style.display = 'none';
                document.getElementById('stopWebcamBtn').style.display = 'inline-block';

                const canvas = document.getElementById('webcamCanvas');
                const ctx = canvas.getContext('2d');

                // Enviar 10 cuadros por segundo (100ms) para no congelar el backend Flask
                webcamInterval = setInterval(() => {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        const formData = new FormData();
                        formData.append('image', blob);
                        formData.append('conf', document.getElementById('confRange').value / 100);

                        fetch('/detect', { method: 'POST', body: formData })
                        .then(r => r.json())
                        .then(data => {
                            if (data.image) {
                                const img = new Image();
                                img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                                img.src = 'data:image/jpeg;base64,' + data.image;
                                updateMetricsAndHistory(data);
                                lastApiResponseData = data;
                                document.getElementById('downloadBtn').style.display = 'inline-block';
                            }
                        }).catch(e => console.log("Frame drop"));
                    }, 'image/jpeg', 0.7); // Compresión 70% optimizada para velocidad
                }, 100);
            })
            .catch(err => alert('No se pudo acceder a la cámara: ' + err.message));
        }

        function stopWebcam() {
            if(currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
                clearInterval(webcamInterval);
            }
            document.getElementById('webcamVideo').style.display = 'none';
            document.getElementById('webcamCanvas').style.display = 'none';
            document.getElementById('startWebcamBtn').style.display = 'inline-block';
            document.getElementById('stopWebcamBtn').style.display = 'none';
        }

        function downloadJsonReport() {
            if(!lastApiResponseData) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastApiResponseData, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", `reporte_trafico_${Date.now()}.json`);
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se recibió ninguna imagen'}), 400

        file = request.files['image']
        confianza_minima = float(request.form.get('conf', 0.5)) # Recibe el slider de la interfaz

        if file.filename == '':
            return jsonify({'error': 'Archivo sin nombre'}), 400

        # Leer la imagen directamente de memoria sin guardar en disco
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Archivo de imagen dañado o inválido'}), 400

        # Redimensionado inteligente para rendimiento óptimo
        h, w = img.shape[:2]
        if h > 1000 or w > 1000:
            scale = min(1000 / h, 1000 / w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Ejecución del Modelo YOLO
        resultados = modelo(img, conf=confianza_minima, verbose=False)
        detecciones = resultados[0]

        detections_data = []
        confianzas = []

        for caja in detecciones.boxes:
            clase_id = int(caja.cls)
            confianza = float(caja.conf)

            if clase_id in CLASES_VEHICULOS:
                meta = CLASES_VEHICULOS[clase_id]
                nombre_clase = meta["name"]
                color_clase = meta["color"]

                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                detections_data.append({
                    'class': nombre_clase,
                    'confidence': confianza * 100,
                    'coords': [x1, y1, x2, y2]
                })
                confianzas.append(confianza * 100)

                # Dibujo del Bounding Box Dinámico personalizado
                cv2.rectangle(img, (x1, y1), (x2, y2), color_clase, 2)

                # Fondo para el texto (Mejora la legibilidad)
                label = f"{nombre_clase} {confianza*100:.1f}%"
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - h_txt - 8), (x1 + w_txt + 4, y1), color_clase, -1)

                # Texto en blanco sobre el fondo de color
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Codificación a Base64 para enviar a la interfaz
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_base64 = base64.b64encode(buffer).decode()

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'image': img_base64,
            'total_detections': len(detections_data),
            'avg_confidence': sum(confianzas) / len(confianzas) if confianzas else 0,
            'detections': detections_data
        })

    except Exception as e:
        return jsonify({'error': f'Excepción interna: {str(e)}'}), 500

if __name__ == '__main__':
    # Inicialización limpia
    print("\n" + "="*50)
    print("🚗 SISTEMA DE DETECCIÓN VEHICULAR INICIADO")
    print("🔗 Abre en tu navegador: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
