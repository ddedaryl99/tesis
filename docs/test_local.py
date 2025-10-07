from ultralytics import YOLO

# 1️⃣ Cargar tu modelo entrenado
m = YOLO(r"C:\Users\dviva\OneDrive\Desktop\Test 2025\best.pt")

# 3️⃣ Realizar predicción en cámara
m.predict(
    source=0,          # 0 = cámara integrada, 1 = cámara USB
    show=True,         # mostrar en ventana
    save=True,         # guardar los resultados
    conf=0.53,         # confianza mínima
    imgsz=640,         # tamaño de entrada
    project=r"C:\Users\dviva\OneDrive\Desktop\scripts\runs\detect",
    name="combined_mc_grouped",
    exist_ok=True
)
