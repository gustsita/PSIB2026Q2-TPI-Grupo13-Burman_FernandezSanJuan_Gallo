"""Entrena el clasificador final con las 647 lesiones de BUSI y lo guarda en modelos/.

Antes de entrenar lo evalúa con validación cruzada agrupada por duplicados (como la Parte 6),
y guarda esas métricas junto con el modelo. Tarda unos minutos.

Uso (desde la raíz del repo):  python scripts/entrenar_modelo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from busi import descriptores, modelo, segmentacion
from busi.config import load_config
from busi.datos import cargar_caso, cargar_imagen, listar_casos

UMBRAL_DUPLICADOS = 0.4          # el elegido en la Parte 6


def fila_caso(id_caso, caso):
    warnings.filterwarnings("ignore", category=FutureWarning)
    img, mascara = cargar_caso(caso)
    semilla = segmentacion.semilla_desde_mascara(mascara)      # simula el clic del usuario
    *_, valores = modelo.analizar_lesion(img, semilla)
    return None if valores is None else {"id": id_caso, "clase": caso["clase"], **valores}


casos = listar_casos().set_index("id")
con_lesion = casos[casos["clase"] != "normal"]

print(f"Calculando descriptores de {len(con_lesion)} lesiones...")
filas = Parallel(n_jobs=-1)(delayed(fila_caso)(i, c) for i, c in con_lesion.iterrows())
tabla = pd.DataFrame([f for f in filas if f is not None]).set_index("id")
columnas = [c for c in tabla.columns if c != "clase"]
y = (tabla["clase"] == "malignant").astype(int).to_numpy()
print(f"{len(tabla)} lesiones con segmentación válida, {len(columnas)} descriptores")

huellas = np.array([descriptores.huella(cargar_imagen(r)) for r in casos["ruta"]])
grupos = pd.Series(descriptores.grupos_duplicados(descriptores.matriz_correlacion(huellas), UMBRAL_DUPLICADOS),
                   index=casos.index)

por_pliegue, prob = descriptores.evaluar_clasificador(tabla[columnas], y, grupos.loc[tabla.index],
                                                      seed=load_config()["SEED"])
metricas = {"auc": float(roc_auc_score(y, prob)),
            "sensibilidad": float(por_pliegue["sensibilidad"].mean()),
            "especificidad": float(por_pliegue["especificidad"].mean()),
            "lesiones": len(tabla)}
print("Validación cruzada agrupada por duplicados:")
print(f"  AUC {metricas['auc']:.3f}  sensibilidad {metricas['sensibilidad']:.2f}  especificidad {metricas['especificidad']:.2f}")

referencia = tabla.groupby("clase")[columnas].median().to_dict("index")     # valores típicos de cada clase
modelo.guardar(modelo.entrenar(tabla[columnas], y), columnas, metricas, referencia)
print(f"Modelo guardado en {modelo.ARCHIVO_MODELO}")
