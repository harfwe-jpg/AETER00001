import cv2
from ultralytics import YOLO
import time
import sys
import os
from datetime import datetime

# Cargar el modelo YOLOv8 oficial
modelo = YOLO("yolov8n.pt")

# Clases con sus respectivos colores en formato BGR
CLASES_VEHICULOS = {
    2: {"name": "auto", "color": (0, 255, 0)},       # Verde
    3: {"name": "motocicleta", "color": (255, 0, 0)}, # Azul
    5: {"name": "autobus", "color": (0, 165, 255)},   # Naranja
    7: {"name": "camion", "color": (0, 0, 255)}       # Rojo
}

def dibujar_caja_estilizada(img, x1, y1, x2, y2, clase, id_objeto, color):
    """Dibuja una caja de texto con fondo sólido e incluye el ID del objeto"""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Crear etiqueta con el ID de rastreo único
    etiqueta = f"ID:{id_objeto} {clase.upper()}"
    (ancho_txt, alto_txt), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Fondo del texto y texto en blanco
    cv2.rectangle(img, (x1, y1 - alto_txt - 8), (x1 + ancho_txt + 6, y1), color, -1)
    cv2.putText(img, etiqueta, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def detectar_en_imagen(ruta_imagen):
    """Detecta vehículos en una imagen estática"""
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"❌ Error: No se pudo leer {ruta_imagen}")
        return

    resultados = modelo(img, conf=0.4, verbose=False)
    detecciones = resultados[0]
    conteos = {}

    for caja in detecciones.boxes:
        clase_id = int(caja.cls)
        confianza = float(caja.conf)

        if clase_id in CLASES_VEHICULOS:
            info = CLASES_VEHICULOS[clase_id]
            x1, y1, x2, y2 = map(int, caja.xyxy[0])

            conteos[info["name"]] = conteos.get(info["name"], 0) + 1
            dibujar_caja_estilizada(img, x1, y1, x2, y2, info["name"], "N/A", info["color"])

    print(f"\n📊 Resumen de Detección en Imagen:")
    for k, v in conteos.items():
        print(f"  - {k.upper()}: {v}")

    cv2.imwrite("resultado_imagen.jpg", img)
    cv2.imshow("Deteccion - Imagen (Cualquier tecla para salir)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def procesar_video_o_camara(fuente, es_webcam=False):
    """Procesador con Tracking Inteligente (ByteTrack) y Captura de Evidencias"""
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la fuente de video: {fuente}")
        return

    out = None
    if not es_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('resultado_video.mp4', fourcc, 30, (640, 480))

    # Control de Conteo Avanzado por ID de Trayectoria
    linea_y = 320
    ids_que_cruzaron = set()  # Almacena los IDs únicos que ya pasaron la línea

    # Diccionario para almacenar el conteo detallado por tipo de vehículo
    reporte_vehicular = {"auto": 0, "motocicleta": 0, "autobus": 0, "camion": 0}

    # Crear carpeta para guardar capturas de los vehículos que cruzan
    os.makedirs("capturas_trafico", exist_ok=True)

    prev_time = 0
    print("\n🎬 Ejecutando sistema con tracking inteligente. Presiona 'q' para salir...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Calcular FPS reales
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Dibujar Línea de Conteo Virtual
        cv2.line(frame, (0, linea_y), (640, linea_y), (0, 165, 255), 2)
        cv2.putText(frame, "PUNTO DE CONTROL CONTROL", (10, linea_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Inferencia con TRACKING ACTIVADO (Usa ByteTrack de forma nativa)
        resultados = modelo.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)
        detecciones = resultados[0]

        vehiculos_en_frame = 0

        # Validamos que existan cajas y que tengan IDs asignados por el tracker
        if detecciones.boxes is not None and detecciones.boxes.id is not None:
            cajas = detecciones; boxes,xyxy.cpu().numpy()
            ids = detecciones.boxes.id.cpu().numpy().astype(int)
            clases = detecciones.boxes.cls.cpu().numpy().astype(int)

            for box, id_obj, clase_id in zip(cajas, ids, clases):
                if clase_id in CLASES_VEHICULOS:
                    vehiculos_en_frame += 1
                    info = CLASES_VEHICULOS[clase_id]
                    x1, y1, x2, y2 = map(int, box)

                    # Obtener el centro geométrico del vehículo
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 4, info["color"], -1)

                    # LÓGICA DE CONTEO INMUTABLE: Si el ID no ha cruzado antes y pasa el umbral de la línea
                    if id_obj not in ids_que_cruzaron:
                        # Si el vehículo está cruzando la franja de la línea
                        if abs(cy - linea_y) < 8:
                            ids_que_cruzaron.add(id_obj)
                            reporte_vehicular[info["name"]] += 1

                            # GUARDAR CAPTURA DE PANTALLA COMO EVIDENCIA
                            ahora = datetime.now().strftime("%H%M%S")
                            cv2.imwrite(f"capturas_trafico/id_{id_obj}_{info['name']}_{ahora}.jpg", frame)

                    # Dibujar bounding box estilizado con su ID único real
                    dibujar_caja_estilizada(frame, x1, y1, x2, y2, info["name"], id_obj, info["color"])

        # HUD / Panel de Control Superior Izquierdo (Opaco)
        cv2.rectangle(frame, (5, 5), (250, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"En pantalla: {vehiculos_en_frame}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Total Detectados: {len(ids_que_cruzaron)}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Detector Vehicular Pro con ByteTrack", frame)
        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # REPORTE DE ANALÍTICA FINAL EN CONSOLA
    print(f"\n{'='*45}")
    print("📊 INFORME ESTADÍSTICO DE FLUIDEZ VIAL")
    print(f"{'='*45}")
    print(f"🚗 Autos cruzados      : {reporte_vehicular['auto']}")
    print(f"🏍️ Motocicletas        : {reporte_vehicular['motocicleta']}")
    print(f"🚌 Autobuses            : {reporte_vehicular['autobus']}")
    print(f"🚚 Camiones de carga   : {reporte_vehicular['camion']}")
    print(f"{'-'*45}")
    print(f"📈 TOTAL VEHÍCULOS ÚNICOS: {len(ids_que_cruzaron)}")
    print(f"📁 Imágenes de evidencia guardadas en: /capturas_trafico")
    print(f"{'='*45}\n")

if __name__ == "__main__":
    print("="*50)
    print("🚗 INICIANDO SOFTWARE DE CONTROL DE TRÁFICO")
    print("="*50)

    if len(sys.argv) > 1:
        entrada = sys.argv[1]
        if entrada.lower() in ['webcam', 'cam', '0']:
            procesar_video_o_camara(0, es_webcam=True)
        elif entrada.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            procesar_video_o_camara(entrada, es_webcam=False)
        elif entrada.lower().endswith(('.jpg', '.jpeg', '.png')):
            detectar_en_imagen(entrada)
        else:
            print("❌ Formato o comando no reconocido.")
    else:
        # Por defecto, si ejecutas sin argumentos, buscará la webcam para facilitar tu test
        procesar_video_o_camara(0, es_webcam=True)
