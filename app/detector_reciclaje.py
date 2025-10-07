import torch, cv2

print("🔄 Cargando modelo YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'C:\Users\dviva\OneDrive\Desktop\Test 2025\best.pt',
                       source='github')
print("✅ Modelo cargado correctamente.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ No se pudo abrir la cámara.")
    exit()

print("🎥 Cámara activa. Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)
    cv2.imshow("♻️ Detección de Reciclaje", results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
print("🛑 Detección finalizada.")
