import sys
import pathlib
import time
import csv
import cv2
import torch
import torch.nn as nn
import torch.serialization
from collections import defaultdict

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
YOLOV5_DIR = r"C:\Users\dviva\yolov5"        # carpeta donde está YOLOv5
MODEL_PATH = r"C:\Users\dviva\yolov5\best.torchscript" # modelo entrenado
STABILITY_TIME = 4.0                          # segundos mínimos para confirmar detección
CSV_FILE = "detecciones.csv"                  # archivo CSV de salida
CAMERA_INDEX = 0                              # índice de cámara
CONF_THRESHOLD = 0.25                         # confianza mínima
IMG_SIZE = 640                                # tamaño de imagen

# =========================================================
# AJUSTES DE COMPATIBILIDAD Y CARGA DEL MODELO
# =========================================================
sys.path.insert(0, YOLOV5_DIR)
pathlib.PosixPath = pathlib.WindowsPath  # Fix modelos entrenados en Linux

# Importaciones del core YOLOv5
from models.common import DetectMultiBackend, Conv, C3, Bottleneck, SPPF
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.yolo import DetectionModel

# 🔒 Permitir todas las clases necesarias para PyTorch ≥ 2.6
torch.serialization.add_safe_globals([
    nn.Sequential,
    nn.SiLU,
    nn.BatchNorm2d,
    nn.Conv2d,
    DetectionModel,
    Conv,
    C3,
    Bottleneck,
    SPPF,  # ✅ agregado: capa final de YOLOv5
])

# =========================================================
# CARGA DEL MODELO
# =========================================================
print("🔄 Cargando modelo YOLOv5 local...")

device = select_device('cpu')
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(IMG_SIZE, s=stride)

if not names or len(names) == 0:
    names = ['basura', 'carton', 'papel', 'plastico', 'reciclable', 'vidrio']

print("✅ Modelo cargado correctamente.")
print("Clases del modelo:", names)

# =========================================================
# INICIALIZAR CÁMARA
# =========================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    sys.exit()

cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

# Crear archivo CSV si no existe
with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Clase", "Cantidad_confirmada"])

print("🎥 Iniciando detección... Presiona 'q' para salir.")

# =========================================================
# VARIABLES DE CONTROL
# =========================================================
detecciones_confirmadas = defaultdict(int)
deteccion_activa = None
inicio_deteccion = 0

# =========================================================
# BUCLE PRINCIPAL
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar a 640x640 (igual que entrenamiento)
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    im_tensor = torch.from_numpy(img).to(device)
    im_tensor = im_tensor.permute(2, 0, 1).float() / 255.0
    if im_tensor.ndimension() == 3:
        im_tensor = im_tensor.unsqueeze(0)

    # Predicción
    pred = model(im_tensor)
    pred = non_max_suppression(pred, CONF_THRESHOLD, 0.45, classes=None, agnostic=False)

    # Dibujar detecciones
    for det in pred:
        annotator = Annotator(frame_resized, line_width=2, example=str(names))
        if len(det):
            det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame_resized.shape).round()

            # Detección más confiable
            confs = det[:, 4]
            idx = torch.argmax(confs).item()
            clase_id = int(det[idx, 5])
            clase = names[clase_id]
            conf = float(confs[idx])

            if conf > CONF_THRESHOLD:
                cv2.putText(frame_resized, f"{clase} ({conf:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Lógica de detección estable
                if deteccion_activa == clase:
                    duracion = time.time() - inicio_deteccion
                    if duracion >= STABILITY_TIME:
                        detecciones_confirmadas[clase] += 1
                        print(f"✅ Confirmado: {clase} ({detecciones_confirmadas[clase]})")

                        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([clase, detecciones_confirmadas[clase]])

                        deteccion_activa = None
                        inicio_deteccion = 0
                else:
                    deteccion_activa = clase
                    inicio_deteccion = time.time()

                # Dibujar caja y etiqueta
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        frame_resized = annotator.result()

    cv2.imshow("🟢 Detección estable YOLOv5", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# FINALIZACIÓN
# =========================================================
cap.release()
cv2.destroyAllWindows()

print("\n📦 Registro final:")
for clase, cantidad in detecciones_confirmadas.items():
    print(f"  - {clase}: {cantidad} veces confirmada")

print(f"\nResultados guardados en: {CSV_FILE}")
