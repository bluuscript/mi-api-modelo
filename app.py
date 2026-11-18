import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Nombre exacto del archivo del modelo que subiste a GitHub
MODELO = "cifar_cnn_int8.onnx"

# Las 10 clases de CIFAR-10, en el mismo orden del notebook
CLASES = ["avión", "auto", "pájaro", "gato", "ciervo",
          "perro", "rana", "caballo", "barco", "camión"]

# Misma normalización que se usó al entrenar el modelo
MEDIA = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)

# El modelo se abre una sola vez, al arrancar el servicio
sesion = ort.InferenceSession(MODELO, providers=["CPUExecutionProvider"])

app = FastAPI(title="API de mi modelo")


@app.get("/")
def inicio():
    return {"mensaje": "Mi API de clasificación de imágenes está funcionando. Prueba /docs"}


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.post("/predict")
async def predict(archivo: UploadFile = File(...)):
    datos = await archivo.read()
    try:
        img = Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida")

    img = img.resize((32, 32))
    x = np.asarray(img, dtype=np.float32) / 255.0   # valores entre 0 y 1
    x = x.transpose(2, 0, 1)                        # de (alto, ancho, canal) a (canal, alto, ancho)
    x = ((x - MEDIA) / STD)[None].astype(np.float32)  # normalizar y agregar la dimensión del lote

    logits = sesion.run(None, {"imagen": x})[0][0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    i = int(p.argmax())
    return {"clase": CLASES[i], "probabilidad": round(float(p[i]), 3)}
