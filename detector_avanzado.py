import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
from pathlib import Path
import sqlite3
from collections import defaultdict
import time

class DetectorVehicularAvanzado:
    """Detector vehicular con tracking de ID único, análisis de ROI y analítica de tráfico"""

    def __init__(self, modelo_yolo="yolov8n.pt", db_path="vehiculos.db"):
        self.modelo = YOLO(modelo_yolo)
        self.db_path = db_path
        self.clases_vehiculos = {
            2: "Auto",
            3: "Motocicleta",
            5: "Autobús",
            7: "Camión"
        }

        # Paleta de colores consistente (BGR)
        self.colores = {
            "Auto": (0, 255, 0),        # Verde
            "Motocicleta": (255, 0, 0),  # Azul
            "Autobús": (0, 165, 255),    # Naranja
            "Camión": (0, 0, 255)        # Rojo
        }

        self._crear_db()
        self.estadisticas = defaultdict(int)
        self.confianzas = defaultdict(list)

    def _crear_db(self):
        """Crea la estructura relacional indexada en SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla de detecciones únicas (se añade la columna id_objeto para el tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detecciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                archivo TEXT,
                id_objeto INTEGER,
                tipo_vehiculo TEXT,
                confianza REAL,
                x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sesiones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                archivo TEXT,
                total_detecciones INTEGER,
                promedio_confianza REAL,
                duracion_segundos REAL
            )
        ''')

        # Índices para acelerar las consultas analíticas de tiempo
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detecciones(timestamp)')
        conn.commit()
        conn.close()

    def _guardar_deteccion(self, archivo, id_objeto, tipo, confianza, coords):
        """Inserta detecciones evitando duplicados del mismo ID de objeto en la sesión"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Verificar si este ID de objeto ya fue registrado para este archivo específico
        cursor.execute('''
            SELECT id FROM detecciones WHERE archivo = ? AND id_objeto = ?
        ''', (archivo, id_objeto))

        if cursor.fetchone() is None:
            x1, y1, x2, y2 = coords
            cursor.execute('''
                INSERT INTO detecciones (archivo, id_objeto, tipo_vehiculo, confianza, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (archivo, id_objeto, tipo, confianza, x1, y1, x2, y2))
            conn.commit()

        conn.close()

    def _guardar_sesion(self, archivo, total, promedio, duracion):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sesiones (archivo, total_detecciones, promedio_confianza, duracion_segundos)
            VALUES (?, ?, ?, ?)
        ''', (archivo, total, promedio, duracion))
        conn.commit()
        conn.close()

    def _calcular_centro(self, x1, y1, x2, y2):
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def detectar_imagen(self, ruta_imagen, roi_poligono=None, guardar_resultados=True):
        """Detecta vehículos en una imagen filtrando por una Región de Interés (ROI) opcional"""
        inicio = time.time()
        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"❌ Error: No se pudo leer {ruta_imagen}")
            return None

        nombre_archivo = Path(ruta_imagen).name
        resultados = self.modelo(img, conf=0.4, verbose=False)
        detecciones = resultados[0]

        detecciones_data = []
        self.estadisticas.clear()
        self.confianzas.clear()

        # Dibujar la ROI en la imagen si existe (Lista de tuplas [(x1,y1), (x2,y2)...])
        if roi_poligono is not None:
            pts = np.array(roi_poligono, np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 0), 2)

        for idx, caja in enumerate(detecciones.boxes):
            clase_id = int(caja.cls)
            confianza = float(caja.conf)

            if clase_id in self.clases_vehiculos:
                tipo = self.clases_vehiculos[clase_id]
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                cx, cy = self._calcular_centro(x1, y1, x2, y2)

                # Validar si el centro del vehículo cae dentro de la ROI
                if roi_poligono is not None:
                    dentro_roi = cv2.pointPolygonTest(np.array(roi_poligono, np.int32), (cx, cy), False)
                    if dentro_roi < 0:
                        continue # Ignorar vehículo fuera del área

                self.estadisticas[tipo] += 1
                self.confianzas[tipo].append(confianza * 100)

                detecciones_data.append({
                    'id': idx, 'tipo': tipo, 'confianza': round(confianza * 100, 2), 'coords': (x1, y1, x2, y2)
                })

                color = self.colores.get(tipo, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{tipo} {confianza*100:.1f}%", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                self._guardar_deteccion(nombre_archivo, idx, tipo, confianza * 100, (x1, y1, x2, y2))

        duracion = time.time() - inicio
        self._mostrar_resumen(nombre_archivo, duracion)

        if detecciones_data:
            promedios = [c for t in self.confianzas.values() for c in t]
            promedio = sum(promedios) / len(promedios)
            self._guardar_sesion(nombre_archivo, len(detecciones_data), round(promedio, 2), round(duracion, 2))

        if guardar_resultados:
            cv2.imwrite(f"detecciones_{Path(ruta_imagen).stem}.jpg", img)

        return detecciones_data

    def detectar_video(self, ruta_video, procesar_cada_n_frames=3, roi_poligono=None):
        """Procesa video con Tracking basado en Centroides para evitar duplicados en BD"""
        inicio = time.time()
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened(): return

        nombre_archivo = Path(ruta_video).name
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('detecciones_video_pro.mp4', fourcc, fps, (ancho, alto))

        frame_count = 0
        objetos_rastreados = {} # {id_objeto: (cx, cy, tiempo_vida, tipo)}
        siguiente_id_objeto = 0
        ultimo_estado_boxes = [] # Corrige el parpadeo de fotogramas vacíos

        self.estadisticas.clear()
        self.confianzas.clear()

        print(f"\n🎬 Ejecutando Analítica sobre: {nombre_archivo}")

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Dibujar área de interés si está definida
            if roi_poligono is not None:
                cv2.polylines(frame, [np.array(roi_poligono, np.int32)], True, (255, 255, 0), 2)

            # Inferencia selectiva por salto de frames
            if frame_count % procesar_cada_n_frames == 0:
                frame_redimensionado = cv2.resize(frame, (640, 480))
                resultados = self.modelo(frame_redimensionado, conf=0.4, verbose=False)
                detecciones = resultados[0]

                centros_actuales = []
                ultimo_estado_boxes.clear()

                for caja in detecciones.boxes:
                    clase_id = int(caja.cls)
                    confianza = float(caja.conf)

                    if clase_id in self.clases_vehiculos:
                        tipo = self.clases_vehiculos[clase_id]
                        x1, y1, x2, y2 = map(int, caja.xyxy[0])

                        # Escalamiento inverso
                        x1, x2 = int(x1 * (ancho / 640)), int(x2 * (ancho / 640))
                        y1, y2 = int(y1 * (alto / 480)), int(y2 * (alto / 480))
                        cx, cy = self._calcular_centro(x1, y1, x2, y2)

                        if roi_poligono is not None:
                            if cv2.pointPolygonTest(np.array(roi_poligono, np.int32), (cx, cy), False) < 0:
                                continue

                        centros_actuales.append((cx, cy, tipo, confianza, (x1, y1, x2, y2)))

                # Algoritmo de Tracking Predictivo Minimalista
                nuevos_objetos = {}
                for cx, cy, tipo, conf, coords in centros_actuales:
                    emparejado = False
                    # Buscar el objeto previo más cercano
                    for id_obj, (px, py, vida, t_prev) in objetos_rastreados.items():
                        distancia = np.hypot(cx - px, cy - py)
                        if distancia < 45 and tipo == t_prev: # Umbral de píxeles por movimiento
                            nuevos_objetos[id_obj] = (cx, cy, 10, tipo)
                            emparejado = True
                            self._guardar_deteccion(nombre_archivo, id_obj, tipo, conf * 100, coords)
                            ultimo_estado_boxes.append((coords, tipo, id_obj))
                            break

                    if not emparejado:
                        nuevos_objetos[siguiente_id_objeto] = (cx, cy, 10, tipo)
                        self.estadisticas[tipo] += 1
                        self.confianzas[tipo].append(conf * 100)
                        self._guardar_deteccion(nombre_archivo, siguiente_id_objeto, tipo, conf * 100, coords)
                        ultimo_estado_boxes.append((coords, tipo, siguiente_id_objeto))
                        siguiente_id_objeto += 1

                objetos_rastreados = nuevos_objetos

            # Pintar las cajas persistentes (Evita el parpadeo visual en frames intermedios)
            for coords, tipo, id_obj in ultimo_estado_boxes:
                x1, y1, x2, y2 = coords
                color = self.colores.get(tipo, (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{id_obj} {tipo}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        duracion = time.time() - inicio
        self._mostrar_resumen(nombre_archivo, duracion)

        if self.estadisticas:
            total = sum(self.estadisticas.values())
            promedios = [c for t in self.confianzas.values() for c in t]
            self._guardar_sesion(nombre_archivo, total, round(sum(promedios)/len(promedios), 2), round(duracion, 2))

    def obtener_hora_pico_trafico(self):
        """Analiza analíticamente las marcas de tiempo para extraer la hora con mayor afluencia"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT STRFTIME('%H:00', timestamp) as hora, COUNT(*) as flujo
            FROM detecciones
            GROUP BY hora
            ORDER BY flujo DESC
            LIMIT 1
        ''')
        res = cursor.fetchone()
        conn.close()

        print(f"\n{'='*50}\n📊 REPORTE DE INTELIGENCIA DE TRÁFICO")
        if res:
            print(f"   La hora pico detectada es: {res[0]} con un volumen de {res[1]} registros.")
        else:
            print("   Datos insuficientes para calcular la tendencia horaria.")
        print(f"{'='*50}\n")

    def _mostrar_resumen(self, archivo, duracion):
        print(f"\n{'='*50}\n📊 RESUMEN FINAL: {archivo}\n{'='*50}")
        if not self.estadisticas:
            print("   No se procesaron registros nuevos.")
        else:
            for tipo, cantidad in sorted(self.estadisticas.items(), key=lambda x: x[1], reverse=True):
                prom = sum(self.confianzas[tipo]) / len(self.confianzas[tipo])
                print(f"   • {tipo:12} → {cantidad:3} unidades únicas (Confianza: {prom:.1f}%)")
        print(f"\n⏱️  Tiempo de Cómputo: {duracion:.2f}s\n{'='*50}\n")

    def generar_reporte_json(self, archivo_salida="reporte_analitico.json"):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sesiones ORDER BY timestamp DESC")
        sesiones = [dict(row) for row in cursor.fetchall()]
        conn.close()

        with open(archivo_salida, 'w') as f:
            json.dump({"generado": datetime.now().isoformat(), "historico_sesiones": sesiones}, f, indent=4)
        print(f"✅ Historial maestro exportado a: {archivo_salida}")

# ============== DEMOSTRACIÓN DE EJECUCIÓN PRO ==============
if __name__ == "__main__":
    # Instanciamos el motor analítico
    detector = DetectorVehicularAvanzado()

    # Definimos una Región de Interés ficticia (Polígono de 4 esquinas en una resolución clásica)
    # Solo los autos cuyo centro geométrico toque este espacio serán guardados en SQLite
    mi_zona_interes = [(50, 400), (300, 150), (600, 150), (600, 400)]

    # 1. Procesar video con Tracking e ID único (No duplicará registros por frame)
    # detector.detectar_video("autopista.mp4", procesar_cada_n_frames=3, roi_poligono=mi_zona_interes)

    # 2. Consultar analítica predictiva de horas con mayor volumen vehicular
    detector.obtener_hora_pico_trafico()

    # 3. Exportar reportes consolidados
    detector.generar_reporte_json()
