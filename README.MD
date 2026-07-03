# 🚗 Detector Vehicular con YOLO v8

Una aplicación moderna para detectar vehículos (autos, motos, autobuses, camiones) usando YOLOv8 con opciones en script Python o interfaz web interactiva.

## 📋 Requisitos

- Python 3.8+
- pip (gestor de paquetes)

## 🚀 Instalación

### 1. Clonar o descargar los archivos

```bash
cd tu-carpeta-proyecto
```

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota**: La primera ejecución descargará automáticamente el modelo YOLOv8 (~50MB).

---

## 💻 Opciones de Uso

### Opción 1: Script Básico (`detector_vehicular_basico.py`)

Ideal para scripts de escritorio, análisis de archivos.

#### Detectar en una imagen:
```bash
python detector_vehicular_basico.py imagen.jpg
```

#### Detectar en un video:
```bash
python detector_vehicular_basico.py video.mp4
```

#### Detección en tiempo real (webcam):
```bash
python detector_vehicular_basico.py webcam
```

**Salidas**:
- `detecciones.jpg` - Imagen con bounding boxes
- `detecciones_video.mp4` - Video procesado

---

### Opción 2: Aplicación Web (`detector_web.py`)

Interfaz web moderna con soporte para imagen y webcam.

#### Iniciar servidor:
```bash
python detector_web.py
```

Luego abre tu navegador en: **http://localhost:5000**

#### Características:
- ✨ Interfaz moderna y atractiva
- 📸 Detección en imágenes (drag & drop)
- 🎥 Detección en tiempo real desde webcam
- 📊 Estadísticas de confianza
- 💾 Descarga de resultados

---

## 🎛️ Modelos YOLOv8 Disponibles

Puedes cambiar el tamaño del modelo en el código:

| Modelo | Tamaño | Velocidad | Precisión |
|--------|--------|-----------|-----------|
| `yolov8n` | 3.2MB | ⚡⚡⚡ Rápido | ⭐⭐⭐ |
| `yolov8s` | 22.5MB | ⚡⚡ Medio | ⭐⭐⭐⭐ |
| `yolov8m` | 49.7MB | ⚡ Lento | ⭐⭐⭐⭐ |
| `yolov8l` | 83.7MB | Muy lento | ⭐⭐⭐⭐⭐ |

Para cambiar, edita la línea:
```python
modelo = YOLO("yolov8n.pt")  # Cambiar a yolov8s, yolov8m, etc.
```

---

## 🔧 Configuración Avanzada

### Ajustar confianza de detección

En `detector_vehicular_basico.py` (línea ~51):
```python
resultados = modelo(img, conf=0.5)  # 0.5 = 50% confianza mínima
```

Aumenta a `0.7` para menos falsos positivos, baja a `0.3` para más detecciones.

### Agregar más clases de objetos

Edita el diccionario `CLASES_VEHICULOS`:
```python
CLASES_VEHICULOS = {
    0: "persona",
    2: "auto",
    3: "motocicleta",
    5: "autobús",
    7: "camión",
    # ... más clases
}
```

Ver todas las clases con:
```python
from ultralytics import YOLO
modelo = YOLO("yolov8n.pt")
print(modelo.names)
```

---

## 📊 Ejemplo de Salida

```
🚗 Vehículos detectados: 3
  1. AUTO - Confianza: 95.32%
  2. MOTOCICLETA - Confianza: 87.21%
  3. CAMIÓN - Confianza: 92.15%
✅ Imagen guardada: detecciones.jpg
```

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### Error: "CUDA not available" (puedes ignorarlo)
Se ejecutará en CPU, más lento pero funciona.

### Webcam no funciona
- Verifica permisos de cámara
- En Windows, permite acceso en Configuración > Privacidad
- En Mac, ve a Seguridad > Cámara

### Modelo descarga muy lentamente
```python
# Descargar manualmente antes:
from ultralytics import YOLO
modelo = YOLO("yolov8n.pt")
```

---

## 📁 Estructura de Archivos

```
tu-proyecto/
├── detector_vehicular_basico.py    # Script standalone
├── detector_web.py                 # Aplicación web Flask
├── requirements.txt                # Dependencias
├── README.md                       # Este archivo
└── detecciones/                    # Resultados (se crea automáticamente)
    ├── detecciones.jpg
    └── detecciones_video.mp4
```

---

## 🚀 Próximas Mejoras

- [ ] Historial de detecciones con base de datos
- [ ] Exportar estadísticas en CSV
- [ ] Integración con cámaras IP
- [ ] Alertas en tiempo real
- [ ] Dashboard de analytics

---

## 📝 Licencia

Basado en YOLOv8 de Ultralytics (AGPLv3)

---

## 💡 Tips

1. **Para máxima velocidad**: Usa `yolov8n` con `conf=0.6`
2. **Para máxima precisión**: Usa `yolov8l` con `conf=0.5`
3. **Para produción**: Corre en GPU: `modelo = YOLO("yolov8n.pt").to("cuda")`
4. **Para debugging**: Agrega `verbose=True` en detección

---

## 🤝 Contacto y Soporte

Si encuentras problemas, verifica:
- Versión de Python: `python --version` (debe ser 3.8+)
- Versión de pip: `pip --version`
- Instala actualizado: `pip install --upgrade ultralytics`

¡Diviértete detectando vehículos! 🎉
