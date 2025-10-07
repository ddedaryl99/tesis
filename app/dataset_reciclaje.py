import os
import shutil
import random
from collections import defaultdict

# Ruta base de tu dataset completo
BASE_DIR = r"C:\Users\dviva\OneDrive\Desktop\scripts\versiones\project_yolov5.v3-v0.yolov5pytorch"

# Subcarpetas donde buscar las imágenes
SUBSETS = ["train", "valid"]

# Carpeta destino
DEST_DIR = os.path.join(BASE_DIR, "dataset_vk")
os.makedirs(os.path.join(DEST_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "labels"), exist_ok=True)

# Cantidad específica por clase
# (usa el ID de clase que tengas en tus archivos .txt)
# Asegúrate de que coincidan con el orden de tus clases en data.yaml
cantidad_por_clase = {
    0: 154,  # vidrio
    1: 250,  # reciclable
    2: 250,  # plastico
    3: 204,  # papel
    4: 255,  # carton
    5: 245   # basura
}

# Diccionario: clase -> lista de archivos
clases = defaultdict(list)

# Recorrer etiquetas
for subset in SUBSETS:
    label_path = os.path.join(BASE_DIR, subset, "labels")
    if not os.path.exists(label_path):
        print(f"⚠️ No existe la carpeta: {label_path}")
        continue

    for label_file in os.listdir(label_path):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(label_path, label_file)) as f:
            lines = f.readlines()
            for line in lines:
                clase = int(line.split()[0])
                clases[clase].append((subset, label_file))
                break  # cuenta solo una vez por imagen

# Seleccionar imágenes según cantidades específicas
seleccionadas = set()
for clase, archivos in clases.items():
    cantidad = cantidad_por_clase.get(clase, 0)
    if cantidad == 0:
        continue
    muestra = random.sample(archivos, min(cantidad, len(archivos)))
    seleccionadas.update(muestra)
    print(f"Clase {clase}: {len(muestra)} imágenes seleccionadas")

# Copiar las imágenes y etiquetas seleccionadas
for subset, label_file in seleccionadas:
    img_name = os.path.splitext(label_file)[0] + ".jpg"
    src_img = os.path.join(BASE_DIR, subset, "images", img_name)
    src_lbl = os.path.join(BASE_DIR, subset, "labels", label_file)
    dst_img = os.path.join(DEST_DIR, "images", img_name)
    dst_lbl = os.path.join(DEST_DIR, "labels", label_file)

    if os.path.exists(src_img) and os.path.exists(src_lbl):
        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

print(f"\n✅ Dataset 'vK' creado en '{DEST_DIR}' con {len(seleccionadas)} imágenes totales.")
