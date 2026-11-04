"""Parte 6 – Descriptores de la lesión y análisis benigno vs. maligno.

Descriptores (calculados sobre la máscara manual Y sobre la segmentación propia,
para separar errores de segmentación de la capacidad de los descriptores):
- Forma (BI-RADS): área, relación alto/ancho (orientación), circularidad,
  solidez (área / área del convex hull), excentricidad, irregularidad del contorno.
- Intensidad: media, desvío, ecogenicidad relativa (lesión vs. anillo de tejido),
  sombra acústica posterior (región debajo de la lesión vs. tejido lateral).
- Textura: GLCM dentro de la lesión (4 direcciones, distancias 1 y 4) con cuantización
  fija, y la misma GLCM **relativa** al anillo de tejido de la misma imagen.

Antes de clasificar hay que agrupar las imágenes casi duplicadas de BUSI
(grupos_duplicados) para que nunca queden en entrenamiento y prueba a la vez.

Estadística: Mann-Whitney U por descriptor con corrección de Holm, y regresión logística
con class_weight="balanced" y validación cruzada StratifiedGroupKFold (grupos = duplicados).
"""

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import measure, transform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dominios

PROPIEDADES_GLCM = ("contrast", "homogeneity", "energy", "correlation")
NOMBRES_GLCM = {"contrast": "contraste", "homogeneity": "homogeneidad",
                "energy": "energia", "correlation": "correlacion"}


# ---- Duplicados ------------------------------------------------------------

def huella(img, tam=64, sin_tendencia=True, sigma=4):
    """La imagen reducida a tam × tam y estandarizada (media 0, desvío 1), como vector.

    Reducir borra el speckle y las diferencias chicas de recorte: dos tomas casi iguales
    de la misma escena quedan casi idénticas.
    Con sin_tendencia=True se le resta a la imagen reducida su versión suavizada
    (gaussiano de `sigma` px): se va el patrón "claro arriba, oscuro abajo" que comparten
    todas las ecografías y queda la estructura propia de cada imagen.
    """
    chica = transform.resize(img.astype(float), (tam, tam), anti_aliasing=True)
    if sin_tendencia:
        chica = chica - ndimage.gaussian_filter(chica, sigma)
    v = chica.ravel()
    return (v - v.mean()) / max(v.std(), 1e-12)


def matriz_correlacion(huellas):
    """Correlación de Pearson entre todas las huellas (filas de una matriz n × tam²)."""
    H = np.asarray(huellas)
    return H @ H.T / H.shape[1]


def grupos_duplicados(correlacion, umbral=0.9):
    """Agrupa las imágenes cuya correlación supera `umbral` (componentes conexas del grafo).

    Si A se parece a B y B a C, A, B y C van al mismo grupo aunque A y C no superen el
    umbral entre sí. Devuelve un array con el número de grupo de cada imagen.
    """
    n = len(correlacion)
    grafo = correlacion >= umbral
    grupo = -np.ones(n, dtype=int)
    actual = 0
    for i in range(n):
        if grupo[i] >= 0:
            continue
        pila = [i]
        grupo[i] = actual
        while pila:
            j = pila.pop()
            for k in np.flatnonzero(grafo[j]):
                if grupo[k] < 0:
                    grupo[k] = actual
                    pila.append(k)
        actual += 1
    return grupo


# ---- Descriptores ----------------------------------------------------------

def descriptores_forma(mascara):
    """Dict con área, alto/ancho, circularidad, solidez, excentricidad e irregularidad.

    - alto_ancho: alto / ancho del rectángulo que contiene la lesión. > 1 es "más alta
      que ancha", rasgo de malignidad en BI-RADS (la lesión crece atravesando los planos
      del tejido, que son horizontales).
    - circularidad: 4π·área / perímetro² (1 = círculo; baja si el borde es irregular).
    - solidez: área / área del convex hull (1 = convexa; baja si tiene entrantes).
    - irregularidad: desvío de la distancia del contorno al centroide, dividido por su
      media. 0 para un círculo; crece con lóbulos y espículas (márgenes "no circunscriptos").
    """
    mascara = mascara.astype(bool)
    props = measure.regionprops(mascara.astype(np.uint8))[0]
    alto = props.bbox[2] - props.bbox[0]
    ancho = props.bbox[3] - props.bbox[1]
    contorno = measure.find_contours(np.pad(mascara, 1).astype(float), 0.5)
    contorno = max(contorno, key=len) - 1                          # el contorno exterior
    radios = np.hypot(contorno[:, 0] - props.centroid[0], contorno[:, 1] - props.centroid[1])
    return {
        "area": props.area,
        "alto_ancho": alto / ancho,
        "circularidad": 4 * np.pi * props.area / props.perimeter ** 2,
        "solidez": props.solidity,
        "excentricidad": props.eccentricity,
        "irregularidad": radios.std() / radios.mean(),
    }


def region_sombra(mascara, profundidad=None):
    """Regiones para medir la sombra acústica posterior.

    - debajo: franja de igual ancho que la lesión (columnas de su bbox), que empieza en el
      borde inferior de la lesión y tiene `profundidad` filas (por defecto, el alto de la
      lesión);
    - lateral: dos franjas de igual tamaño a la izquierda y a la derecha de la anterior,
      a la **misma profundidad**: el tejido con el que se compara, que atravesó la misma
      cantidad de tejido y recibe la misma ganancia del equipo.
    Las regiones se recortan a la imagen. Devuelve (debajo, lateral), máscaras booleanas.
    """
    alto_img, ancho_img = mascara.shape
    filas, cols = np.nonzero(mascara)
    y0, y1, x0, x1 = filas.min(), filas.max() + 1, cols.min(), cols.max() + 1
    prof = profundidad or (y1 - y0)
    ancho = x1 - x0
    debajo = np.zeros(mascara.shape, bool)
    lateral = np.zeros(mascara.shape, bool)
    filas_sombra = slice(y1, min(y1 + prof, alto_img))
    debajo[filas_sombra, x0:x1] = True
    lateral[filas_sombra, max(x0 - ancho, 0):x0] = True
    lateral[filas_sombra, x1:min(x1 + ancho, ancho_img)] = True
    return debajo & ~mascara, lateral & ~mascara


def descriptores_intensidad(img, mascara, excluir=None):
    """Media, desvío, ecogenicidad relativa y sombra acústica posterior.

    - media, desvio: niveles de gris dentro de la lesión.
    - ecogenicidad: media de la lesión / media del anillo de tejido (dominios.anillo, de 5
      a 20 px por fuera, el mismo de las Partes 3 y 4). < 1 = hipoecoica.
    - sombra: media debajo de la lesión / media del tejido lateral a la misma profundidad
      (region_sombra). < 1 = sombra acústica posterior (rasgo de malignidad); > 1 =
      refuerzo posterior (típico de los quistes, que dejan pasar el sonido). NaN si la
      lesión llega al borde de abajo de la imagen.
    `excluir`: píxeles que no se usan (por ejemplo, las marcas reconstruidas por inpainting).
    """
    img = img.astype(float)
    validos = np.ones(img.shape, bool) if excluir is None else ~excluir
    lesion = mascara & validos
    anillo = dominios.anillo(mascara) & validos
    debajo, lateral = region_sombra(mascara)
    debajo, lateral = debajo & validos, lateral & validos
    media_anillo = img[anillo].mean() if anillo.any() else np.nan
    return {
        "media": img[lesion].mean(),
        "desvio": img[lesion].std(),
        "ecogenicidad": img[lesion].mean() / max(media_anillo, 1.0),
        "sombra": (img[debajo].mean() / max(img[lateral].mean(), 1.0)
                   if debajo.sum() >= 20 and lateral.sum() >= 20 else np.nan),
    }


def descriptores_textura(img, mascara, niveles=32, distancias=(1, 4), excluir=None):
    """GLCM dentro de la lesión y relativa al anillo de tejido de la misma imagen.

    Usa dominios.glcm_region (Parte 3): cuantización **fija** a `niveles` niveles (no
    depende del mínimo y máximo de cada imagen, así los valores son comparables entre
    imágenes), solo los pares de píxeles con los dos píxeles dentro de la región, promedio
    de las 4 direcciones.
    - "{prop}_d{d}": la propiedad en la lesión;
    - "{prop}_d{d}_rel": lesión / anillo. La Parte 2 encontró que el tejido de las malignas
      tiene μ/σ más alto incluso lejos de la lesión (posible efecto de la adquisición): el
      cociente compensa lo que tienen en común la lesión y el tejido de esa imagen.
      En la correlación, que puede ser negativa o cercana a 0, se usa la diferencia.
    """
    img = img.astype(float)          # glcm_region multiplica por los niveles: en uint8 desbordaría
    validos = np.ones(img.shape, bool) if excluir is None else ~excluir
    lesion = dominios.glcm_region(img, mascara & validos, niveles, distancias, PROPIEDADES_GLCM)
    anillo = dominios.glcm_region(img, dominios.anillo(mascara) & validos, niveles, distancias, PROPIEDADES_GLCM)
    resultado = {}
    for clave, valor in lesion.items():
        prop, d = clave.rsplit("_", 1)
        nombre = f"{NOMBRES_GLCM[prop]}_{d}"
        resultado[nombre] = valor
        if prop == "correlation":
            resultado[f"{nombre}_rel"] = valor - anillo[clave]
        else:
            resultado[f"{nombre}_rel"] = valor / anillo[clave] if anillo[clave] > 0 else np.nan
    return resultado


def descriptores_caso(img, mascara, excluir=None):
    """Los tres grupos de descriptores de una lesión, en un solo dict."""
    d = descriptores_forma(mascara)
    d.update(descriptores_intensidad(img, mascara, excluir))
    d.update(descriptores_textura(img, mascara, excluir=excluir))
    return d


def tabla_descriptores(casos, cargar, mascaras=None):
    """Recorre los casos y arma un DataFrame de descriptores (una fila por imagen).

    casos: DataFrame de datos.listar_casos() indexado por id (solo imágenes con lesión).
    cargar: función caso -> (img_limpia, mascara_manual, marcas).
    mascaras: dict {id: máscara} con la segmentación propia. Si es None se usa la lesión
    principal de la máscara manual (segmentacion.lesion_principal).
    Las imágenes en las que la segmentación quedó vacía no tienen fila.
    """
    from .segmentacion import lesion_principal
    filas = []
    for id_caso, caso in casos.iterrows():
        img, mascara, marcas = cargar(caso)
        m = lesion_principal(mascara) if mascaras is None else mascaras[id_caso].astype(bool)
        if m.sum() < 10:
            continue
        fila = {"id": id_caso, "clase": caso["clase"]}
        fila.update(descriptores_caso(img, m, excluir=marcas))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("id")


# ---- Estadística -----------------------------------------------------------

def holm(p_valores):
    """Corrección de Holm-Bonferroni: p-valores ajustados, en el mismo orden.

    Se ordenan los m p-valores de menor a mayor; el i-ésimo (desde 0) se multiplica por
    (m − i) y se fuerza que los ajustados no bajen (máximo acumulado), recortando en 1.
    Controla la probabilidad de algún falso positivo como Bonferroni, pero es menos estricta.
    """
    p = np.asarray(p_valores, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    ajustados = np.empty(m)
    acumulado = 0.0
    for i, j in enumerate(orden):
        acumulado = max(acumulado, min((m - i) * p[j], 1.0))
        ajustados[j] = acumulado
    return ajustados


def test_mann_whitney(tabla, columna_clase="clase"):
    """Mann-Whitney U (benign vs malignant) para cada descriptor, con corrección de Holm.

    Devuelve un DataFrame con la mediana de cada clase, el AUC del descriptor solo
    (probabilidad de que una maligna al azar tenga un valor mayor que una benigna; 0,5 =
    no separa) y los p-valores sin corregir y corregidos por Holm.
    """
    descriptores = [c for c in tabla.columns if c != columna_clase]
    filas = []
    for d in descriptores:
        b = tabla.loc[tabla[columna_clase] == "benign", d].dropna()
        m = tabla.loc[tabla[columna_clase] == "malignant", d].dropna()
        u = stats.mannwhitneyu(m, b)
        filas.append({"descriptor": d, "mediana_benignas": b.median(), "mediana_malignas": m.median(),
                      "auc": u.statistic / (len(m) * len(b)), "p": u.pvalue})
    res = pd.DataFrame(filas).set_index("descriptor")
    res["p_holm"] = holm(res["p"])
    return res


def evaluar_clasificador(X, y, grupos, n_splits=5, seed=13):
    """Regresión logística con validación cruzada agrupada y estratificada.

    - StratifiedGroupKFold: cada pliegue tiene una proporción de malignas parecida a la
      del total, y todas las imágenes de un mismo grupo de duplicados caen en el mismo pliegue.
    - Pipeline: imputar faltantes con la mediana (la sombra puede faltar) -> estandarizar
      -> LogisticRegression(class_weight="balanced"), que pesa más a la clase minoritaria
      (malignas) para que el modelo no gane "diciendo siempre benigna".
    Todo se ajusta solo con el pliegue de entrenamiento.

    y: 1 = maligna. Devuelve (por_pliegue, probabilidades): un DataFrame con sensibilidad,
    especificidad y AUC de cada pliegue y la probabilidad de maligna fuera de muestra de
    cada imagen.
    """
    X, y, grupos = np.asarray(X, float), np.asarray(y), np.asarray(grupos)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.full(len(y), np.nan)
    filas = []
    for k, (entrena, prueba) in enumerate(cv.split(X, y, grupos)):
        modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(class_weight="balanced", max_iter=2000))
        modelo.fit(X[entrena], y[entrena])
        p = modelo.predict_proba(X[prueba])[:, 1]
        prob[prueba] = p
        pred = p >= 0.5
        filas.append({"pliegue": k + 1,
                      "sensibilidad": (pred & (y[prueba] == 1)).sum() / (y[prueba] == 1).sum(),
                      "especificidad": (~pred & (y[prueba] == 0)).sum() / (y[prueba] == 0).sum(),
                      "auc": roc_auc_score(y[prueba], p)})
    return pd.DataFrame(filas).set_index("pliegue"), prob
