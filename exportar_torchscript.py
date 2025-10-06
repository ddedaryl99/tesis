from ultralytics import YOLO

# Ruta a tu modelo entrenado
modelo_path = r"C:\Users\dviva\yolov5\best.pt"

# Cargar el modelo
model = YOLO(modelo_path)

# Exportar a formato TorchScript
model.export(format="torchscript")

print("✅ Conversión completa. Archivo generado: best.torchscript.pt")
