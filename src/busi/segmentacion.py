"""Parte 5 – Segmentación de la lesión.

Se segmenta sobre los mapas elegidos en la Parte 4. Enfoque semiautomático: el
usuario marca una semilla dentro de la lesión.

Mapas (Parte 4, sección 5.6), calculados sobre la imagen limpia (artefactos.limpiar):
- media local: ventana 15 px sobre la mediana 7 × 7 (brillo);
- coeficiente de variación σ/μ: ventana 31 px sobre la imagen sin filtrar (variación relativa);
- gradiente: magnitud de Sobel promediada en 31 px, sin filtrar (cantidad de variación).

Métodos (vistos en clase):
- region growing desde la semilla,
- k-means sobre el vector de mapas (intensidad + textura),
- contornos activos (Chan-Vese, skimage.segmentation.chan_vese),
- umbral de Otsu como referencia automática simple.
Refinamiento: apertura, cierre, relleno de huecos, y quedarse con la componente
conexa que contiene la semilla.
Validación: Dice e IoU contra la máscara manual.

La máscara manual se usa SOLO para simular el clic del usuario (semilla_desde_mascara)
y para evaluar. Nunca para recortar ni guiar el método: la zona de trabajo, si se usa,
es una ventana fija centrada en la semilla (zona_trabajo).
"""

from collections import deque

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation
from sklearn.cluster import KMeans

from . import dominios, mapas

METODOS = ("rg_media", "rg_imagen", "kmeans_3mapas", "kmeans_media", "chan_vese", "otsu")


# ---- Mapas y zona de trabajo ----------------------------------------------

def mapas_segmentacion(img):
    """Los tres mapas elegidos en la Parte 4 y la imagen filtrada con la mediana 7 × 7.

    img: imagen limpia (artefactos.limpiar), uint8.
    Devuelve dict con:
    - "imagen": mediana 7 × 7 (la imagen "original" suavizada, para comparar);
    - "media": media local en 15 px sobre la mediana 7 × 7;
    - "cv": σ/μ local en 31 px sobre la imagen sin filtrar;
    - "gradiente": Sobel promediado en 31 px sobre la imagen sin filtrar.
    Son las mismas cuentas que mapas.calcular_mapas, sin la GLCM (que acá no se usa).
    """
    img = img.astype(float)
    mediana = dominios.filtrar(img, "mediana", 2)            # σ = 2 -> ventana 7 × 7
    media15, _ = mapas.mapa_media_varianza(mediana, 15)
    media31, varianza31 = mapas.mapa_media_varianza(img, 31)
    return {
        "imagen": mediana,
        "media": media15,
        "cv": np.sqrt(varianza31) / np.maximum(media31, 1.0),
        "gradiente": ndimage.uniform_filter(filters.sobel(img), 31),
    }


def zona_trabajo(shape, semilla, lado=None):
    """Slices de un cuadrado de `lado` px centrado en la semilla (recortado a la imagen).

    Con lado=None (o 0) devuelve la imagen entera. El tamaño es fijo: no depende de la lesión.
    """
    if not lado:
        return (slice(0, shape[0]), slice(0, shape[1]))
    h = lado // 2
    y, x = semilla
    return (slice(max(y - h, 0), min(y + h + 1, shape[0])),
            slice(max(x - h, 0), min(x + h + 1, shape[1])))


# ---- Métodos ---------------------------------------------------------------

def region_growing(mapa, semilla, tolerancia):
    """Crece desde `semilla` (fila, col) agregando vecinos con |valor − media_región| <= tolerancia.

    Recorrido en anchura (BFS) con una cola y vecindad de 8:
    1. La región empieza con la semilla; su media es el valor de la semilla.
    2. Se saca un píxel de la cola y se miran sus 8 vecinos. Cada vecino se examina
       **una sola vez** (la primera vez que se lo alcanza): si su valor está a menos
       de `tolerancia` de la media actual de la región, entra a la región y a la cola,
       y la media se actualiza.
    3. Termina cuando la cola queda vacía.

    Como la media se actualiza con cada píxel nuevo, la región se va "adaptando": si
    la semilla cayó en un píxel un poco más claro que el resto de la lesión, la media
    baja enseguida hacia la de la lesión.

    Se trabaja con listas de Python (y no con el array) porque acceder a un elemento
    de una lista es mucho más rápido que a un elemento de numpy.
    Devuelve una máscara booleana del tamaño de `mapa`.
    """
    alto, ancho = mapa.shape
    valores = mapa.astype(float).ravel().tolist()
    examinado = bytearray(alto * ancho)
    region = np.zeros(alto * ancho, dtype=bool)

    inicio = semilla[0] * ancho + semilla[1]
    examinado[inicio] = 1
    region[inicio] = True
    suma, n = valores[inicio], 1
    cola = deque([inicio])
    vecinos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while cola:
        p = cola.popleft()
        y, x = divmod(p, ancho)
        for dy, dx in vecinos:
            yy, xx = y + dy, x + dx
            if 0 <= yy < alto and 0 <= xx < ancho:
                q = yy * ancho + xx
                if examinado[q]:
                    continue
                examinado[q] = 1
                if abs(valores[q] - suma / n) <= tolerancia:
                    region[q] = True
                    suma += valores[q]
                    n += 1
                    cola.append(q)
    return region.reshape(alto, ancho)


def kmeans_mapas(mapas_lista, k=2, seed=13):
    """Clusteriza píxeles usando como features los mapas apilados (estandarizados).

    mapas_lista: lista de arrays 2D del mismo tamaño. Cada mapa se **estandariza**
    (se le resta su media y se divide por su desvío) para que todos pesen igual: si no,
    el gradiente (valores de decenas) o la media (de 0 a 255) dominarían a σ/μ (~0,1–1).
    Cada píxel es un punto con len(mapas_lista) coordenadas; k-means lo asigna al
    centro más cercano. Devuelve la imagen de etiquetas (0 .. k−1).
    """
    X = np.stack([m.astype(float) for m in mapas_lista], axis=-1).reshape(-1, len(mapas_lista))
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
    return km.labels_.reshape(mapas_lista[0].shape)


def chan_vese(mapa, semilla, mu=0.05, radio_inicial=20, max_iter=300):
    """Contorno activo de Chan-Vese sobre `mapa` (skimage.segmentation.chan_vese).

    Chan-Vese divide la imagen en dos regiones (adentro y afuera de un contorno) buscando
    que cada una sea lo más **uniforme** posible alrededor de su media, con un costo
    `mu` por el largo del contorno (mu alto = contorno más liso). No necesita bordes
    nítidos: separa por diferencia de medias, que es justo lo que distingue la media local.

    El contorno se representa con una función de nivel φ (positiva adentro, negativa
    afuera) y arranca como un círculo de `radio_inicial` px alrededor de la semilla (el
    clic del usuario). φ se escala a [−1, 1]: skimage solo deja moverse a los puntos donde
    φ está cerca de 0, así que con φ en píxeles (valores de cientos) el contorno no
    evoluciona. Se hacen siempre `max_iter` iteraciones (tol muy chica): con la tolerancia
    por defecto skimage a veces se detiene en la primera.
    El mapa se lleva a [0, 1] antes. Devuelve la región del resultado que contiene la
    semilla (máscara booleana).
    """
    m = mapa.astype(float)
    m = (m - m.min()) / max(m.max() - m.min(), 1e-12)
    yy, xx = np.indices(m.shape)
    fi = radio_inicial - np.hypot(yy - semilla[0], xx - semilla[1])     # > 0 dentro del círculo
    fi = fi / np.abs(fi).max()
    seg = segmentation.chan_vese(m, mu=mu, max_num_iter=max_iter, tol=1e-6, init_level_set=fi)
    return seg == seg[semilla]


def otsu(mapa, semilla):
    """Umbral de Otsu sobre `mapa`: referencia automática simple, sin parámetros.

    Otsu elige el umbral que mejor separa el histograma en dos grupos (máxima varianza
    entre clases). La lesión es el grupo de la semilla (normalmente el oscuro).
    """
    umbral = filters.threshold_otsu(mapa)
    return (mapa < umbral) == (mapa[semilla] < umbral)


# ---- Refinamiento, semilla y evaluación -----------------------------------

def refinar(mascara, semilla, radio=3):
    """Apertura + cierre + relleno de huecos, y deja solo la componente que contiene la semilla."""
    se = morphology.disk(radio)
    m = morphology.closing(morphology.opening(mascara.astype(bool), se), se)
    m = ndimage.binary_fill_holes(m)
    etiquetas = measure.label(m)
    etiqueta = etiquetas[semilla]
    return etiquetas == etiqueta if etiqueta else np.zeros_like(m)


def semilla_desde_mascara(mascara):
    """Para evaluar sin clics: punto interior de la máscara manual (el más alejado del borde)."""
    dist = ndimage.distance_transform_edt(mascara)
    return np.unravel_index(np.argmax(dist), dist.shape)


def lesion_principal(mascara):
    """La lesión que marca el usuario: componente conexa de la máscara manual que contiene la semilla.

    Algunas máscaras tienen más de una componente (varias lesiones, o manchas sueltas del
    dibujo). Con un solo clic se segmenta una lesión, así que se evalúa contra esa.
    """
    etiquetas = measure.label(mascara, connectivity=2)
    return etiquetas == etiquetas[semilla_desde_mascara(mascara)]


def segmentar(mapas_img, semilla, metodo, lado=None, radio=3, **params):
    """Segmenta una imagen con `metodo` (uno de METODOS) y devuelve la máscara refinada.

    mapas_img: salida de mapas_segmentacion. El método se aplica dentro de la zona de
    trabajo (ventana fija de `lado` px centrada en la semilla, o la imagen entera) y el
    resultado se refina con refinar(). Parámetros de cada método:
    - "rg_media" / "rg_imagen": tolerancia (en niveles de gris), sobre la media local o
      sobre la mediana 7 × 7;
    - "kmeans_3mapas" / "kmeans_media": k (cantidad de clusters);
    - "chan_vese": mu;
    - "otsu": ninguno.
    """
    forma = mapas_img["media"].shape
    zona = zona_trabajo(forma, semilla, lado)
    s = (semilla[0] - zona[0].start, semilla[1] - zona[1].start)       # semilla dentro de la zona
    recorte = {nombre: m[zona] for nombre, m in mapas_img.items()}

    if metodo == "rg_media":
        cruda = region_growing(recorte["media"], s, params["tolerancia"])
    elif metodo == "rg_imagen":
        cruda = region_growing(recorte["imagen"], s, params["tolerancia"])
    elif metodo == "kmeans_3mapas":
        etiquetas = kmeans_mapas([recorte["media"], recorte["cv"], recorte["gradiente"]], params["k"])
        cruda = etiquetas == etiquetas[s]
    elif metodo == "kmeans_media":
        etiquetas = kmeans_mapas([recorte["media"]], params["k"])
        cruda = etiquetas == etiquetas[s]
    elif metodo == "chan_vese":
        cruda = chan_vese(recorte["media"], s, mu=params.get("mu", 0.05))
    elif metodo == "otsu":
        cruda = otsu(recorte["media"], s)
    else:
        raise ValueError(f"Método desconocido: {metodo}")

    resultado = np.zeros(forma, dtype=bool)
    resultado[zona] = refinar(cruda, s, radio)
    return resultado


def evaluar_caso(img, mascara, configuraciones, devolver_mascaras=False):
    """Segmenta una imagen con varias configuraciones y mide Dice e IoU.

    img: imagen limpia (artefactos.limpiar); mascara: máscara manual (unión de todas).
    configuraciones: lista de dicts con "metodo", "lado" y los parámetros del método
    (por ejemplo {"metodo": "rg_media", "lado": 300, "tolerancia": 15}).
    La semilla sale de la máscara manual (simula el clic) y se compara contra la lesión
    que contiene la semilla (lesion_principal). La máscara no se usa para nada más.

    Devuelve una lista de dicts (una fila por configuración) con dice, iou, el área
    segmentada y la de la lesión; si devolver_mascaras=True, también la lista de máscaras.
    """
    semilla = semilla_desde_mascara(mascara)
    gt = lesion_principal(mascara)
    mapas_img = mapas_segmentacion(img)
    filas, mascaras = [], []
    for config in configuraciones:
        params = {k: v for k, v in config.items() if k not in ("metodo", "lado")}
        pred = segmentar(mapas_img, semilla, config["metodo"], lado=config["lado"], **params)
        fila = dict(config)
        fila.update(dice=dice(pred, gt), iou=iou(pred, gt), area_pred=int(pred.sum()), area_gt=int(gt.sum()))
        filas.append(fila)
        mascaras.append(pred)
    return (filas, mascaras) if devolver_mascaras else filas


def dice(pred, gt):
    """Coeficiente de Dice: 2·|A ∩ B| / (|A| + |B|). 1 = coinciden; 0 = no se tocan."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + 1e-12)


def iou(pred, gt):
    """Índice de Jaccard (IoU): |A ∩ B| / |A ∪ B|. Siempre ≤ Dice."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    return np.logical_and(pred, gt).sum() / (np.logical_or(pred, gt).sum() + 1e-12)
