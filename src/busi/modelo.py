"""Clasificador final: de una ecografía y un clic a benigna o maligna.

Junta las partes 1, 5 y 6 en el camino completo que recorre un caso nuevo:
limpiar las marcas -> segmentar desde el clic -> calcular los descriptores -> clasificar.

El modelo se entrena con los descriptores de NUESTRA segmentación y no con los de la máscara
manual: un caso nuevo no trae máscara, y la forma de las máscaras manuales está sesgada por el
dibujo (Parte 6). Por eso el rendimiento esperable es el del sistema completo (AUC ≈ 0,85) y
no el de la máscara manual (0,99).
"""

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import artefactos, descriptores, segmentacion
from .config import REPO_ROOT

ARCHIVO_MODELO = REPO_ROOT / "modelos" / "clasificador_busi.joblib"
SEGMENTACION = {"metodo": "rg_imagen", "lado": 200, "tolerancia": 30}   # mejor método de la Parte 5
AREA_MINIMA = 10                                                        # px, igual que en la Parte 6
UMBRAL = 0.5                                                            # igual que en la Parte 6


def analizar_lesion(img, semilla):
    """Limpia la imagen, segmenta desde la semilla y calcula los descriptores.

    img: imagen original en gris (datos.cargar_imagen). semilla: (fila, columna) dentro de la lesión.
    Devuelve (img_limpia, marcas, mascara, descriptores); descriptores es None si la
    segmentación quedó vacía.
    """
    limpia, marcas = artefactos.limpiar(img)
    mapas = segmentacion.mapas_segmentacion(limpia)
    mascara = segmentacion.segmentar(mapas, tuple(int(v) for v in semilla), **SEGMENTACION)
    if mascara.sum() < AREA_MINIMA:
        return limpia, marcas, mascara, None
    return limpia, marcas, mascara, descriptores.descriptores_caso(limpia, mascara, excluir=marcas)


def entrenar(X, y):
    """Regresión logística sobre todas las imágenes, con el mismo pipeline que evalúa la Parte 6."""
    modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(class_weight="balanced", max_iter=2000))
    return modelo.fit(np.asarray(X, float), np.asarray(y))


def guardar(modelo, columnas, metricas, referencia, archivo=ARCHIVO_MODELO):
    """Guarda el modelo con sus métricas y la mediana de cada descriptor por clase (referencia)."""
    archivo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "columnas": list(columnas), "metricas": metricas,
                 "referencia": referencia}, archivo)


def cargar(archivo=ARCHIVO_MODELO):
    if not archivo.exists():
        raise FileNotFoundError(f"No existe {archivo}. Generarlo con: python scripts/entrenar_modelo.py")
    return joblib.load(archivo)


def predecir(guardado, img, semilla):
    """Clasifica una lesión. Devuelve un dict con las imágenes intermedias, la probabilidad y la clase.

    Sin segmentación válida, probabilidad, clase y descriptores quedan en None. El modelo asume que hay una
    lesión en la semilla: no detecta imágenes normales.
    """
    limpia, marcas, mascara, valores = analizar_lesion(img, semilla)
    resultado = {"imagen_limpia": limpia, "marcas": marcas, "mascara": mascara, "descriptores": valores,
                 "area_px": int(mascara.sum()), "probabilidad_maligna": None, "clase": None}
    if valores is not None:
        fila = np.array([[valores[c] for c in guardado["columnas"]]], float)
        p = float(guardado["modelo"].predict_proba(fila)[0, 1])
        resultado.update(probabilidad_maligna=p, clase="maligna" if p >= UMBRAL else "benigna")
    return resultado
