"""Parte 4 – Mapas de características (antes de definir la segmentación).

Idea: en vez de elegir un método de segmentación a ciegas, calcular mapas píxel a
píxel (con ventana deslizante) y medir con las máscaras manuales cuál separa mejor
la lesión del tejido. La segmentación (Parte 5) se hace sobre los mapas ganadores.

Mapas propuestos:
- intensidad (la imagen de entrada, sin ventana),
- media y varianza local, coeficiente de variación local (σ/μ),
- GLCM local: contraste, homogeneidad, energía, correlación, entropía,
- magnitud del gradiente (promedio local).

Entradas (según la Parte 3): la imagen limpia sin filtrar y la mediana 7×7. Se calculan
los mapas sobre las dos para verificar la recomendación de la Parte 3 (textura sin
filtrar, intensidad y bordes sobre la mediana).
"""

import numpy as np
from scipy import ndimage
from skimage import filters
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import roc_auc_score

from . import dominios

VENTANAS = (7, 15, 31)
MAPAS = ("intensidad", "media", "varianza", "cv", "gradiente",
         "contraste", "homogeneidad", "energia", "correlacion", "entropia")
MAPAS_GLCM = ("contraste", "homogeneidad", "energia", "correlacion", "entropia")

# desplazamientos (fila, columna) de la GLCM a distancia 1 en las 4 direcciones: 0°, 45°, 90°, 135°
DESPLAZAMIENTOS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def reducir_niveles(img, niveles=16):
    """Cuantiza a `niveles` grises para que la GLCM sea manejable (como en la clase)."""
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return np.minimum((img * niveles).astype(np.uint8), niveles - 1)


def mapa_media_varianza(img, ventana=9):
    """Devuelve (media_local, varianza_local)."""
    img = img.astype(float)
    media = ndimage.uniform_filter(img, ventana)
    media2 = ndimage.uniform_filter(img ** 2, ventana)
    return media, np.maximum(media2 - media ** 2, 0)


def glcm_ventana(ventana_img, niveles=16, distancias=(1,), angulos=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Propiedades GLCM de una ventana, con la GLCM **sumada** en las 4 direcciones.

    Es la versión "de a una ventana" con skimage. Se usa para verificar mapas_glcm(),
    que calcula lo mismo en todos los píxeles a la vez.
    """
    P = graycomatrix(ventana_img, distancias, angulos, levels=niveles, symmetric=True)
    P = P.sum(axis=(2, 3)).astype(float)
    P /= P.sum()
    props = {nombre: graycoprops(P[:, :, None, None], prop)[0, 0]
             for nombre, prop in (("contraste", "contrast"), ("homogeneidad", "homogeneity"),
                                  ("energia", "energy"), ("correlacion", "correlation"))}
    p = P[P > 0]
    props["entropia"] = -np.sum(p * np.log2(p))
    return props


def _suma_rectangulo(arr, arriba, abajo, izquierda, derecha):
    """Para cada píxel (y, x), suma de arr[y + arriba : y + abajo + 1, x + izquierda : x + derecha + 1].

    Lo de afuera de la imagen cuenta como 0. Es un filtro de media (uniform_filter, la
    ventana deslizante de la clase de Mejoramiento) multiplicado por el tamaño de la
    ventana, con el origen corrido para que el rectángulo no tenga que estar centrado
    en el píxel.
    """
    n_filas, n_cols = abajo - arriba + 1, derecha - izquierda + 1
    origen = (-(arriba + n_filas // 2), -(izquierda + n_cols // 2))
    media = ndimage.uniform_filter(arr.astype(float), (n_filas, n_cols), mode="constant", origin=origen)
    return media * n_filas * n_cols


def mapas_glcm(img, ventana=15, niveles=16):
    """Mapas GLCM locales: las propiedades de la GLCM de la ventana centrada en cada píxel.

    La GLCM de una ventana cuenta los pares de píxeles vecinos (distancia 1, sumando
    las 4 direcciones y los dos sentidos: GLCM simétrica) según sus niveles de gris (i, j).
    Cada propiedad es un promedio sobre esos pares, así que no hace falta construir la
    matriz en cada píxel: alcanza con **sumas en ventana deslizante** (como un filtro de
    media) sobre una imagen que tiene, en cada píxel p, el valor del par (p, p + desplazamiento):
    - contraste = promedio de (i − j)²
    - homogeneidad = promedio de 1 / (1 + (i − j)²)
    - correlación = (promedio de i·j − μ²) / σ², con μ y σ² la media y varianza de los
      niveles de los pares
    - energía = √(Σ P(i, j)²) y entropía = −Σ P(i, j)·log₂ P(i, j) necesitan la matriz P:
      cada casilla P(i, j) es la fracción de pares de la ventana que valen (i, j), otra suma en ventana.
    Un par cuenta si sus **dos** píxeles caen en la ventana, igual que graycomatrix sobre
    el recorte de la ventana: el resultado es exacto (ver glcm_ventana) y mucho más rápido.
    En los bordes de la imagen solo se cuentan los pares que caen adentro.

    `img` tiene que estar cuantizada (reducir_niveles); `ventana` impar.
    Devuelve dict {propiedad: mapa}.
    """
    q = img.astype(int)
    alto, ancho = q.shape
    h = ventana // 2
    pares = []                       # (a, b, válido, rectángulo) para cada desplazamiento
    for dy, dx in DESPLAZAMIENTOS:
        b = np.full_like(q, -1)
        ys, yd = slice(max(-dy, 0), alto - max(dy, 0)), slice(max(dy, 0), alto - max(-dy, 0))
        xs, xd = slice(max(-dx, 0), ancho - max(dx, 0)), slice(max(dx, 0), ancho - max(-dx, 0))
        b[ys, xs] = q[yd, xd]                        # b(p) = q(p + desplazamiento)
        # posiciones p de la ventana tales que p + desplazamiento también está en la ventana
        rect = (-h + max(-dy, 0), h - max(dy, 0), -h + max(-dx, 0), h - max(dx, 0))
        pares.append((q, b, b >= 0, rect))

    def suma(funcion):
        """Suma en la ventana de funcion(a, b) sobre los pares de las 4 direcciones."""
        return sum(_suma_rectangulo(np.where(v, funcion(a, b), 0.0), *rect) for a, b, v, rect in pares)

    cuenta = np.maximum(suma(lambda a, b: 1.0), 1e-12)     # cantidad de pares en cada ventana
    def promedio(funcion):
        return suma(funcion) / cuenta

    contraste = promedio(lambda a, b: (a - b) ** 2.0)
    homogeneidad = promedio(lambda a, b: 1 / (1 + (a - b) ** 2.0))
    mu = promedio(lambda a, b: (a + b) / 2.0)                    # GLCM simétrica: μ_i = μ_j
    varianza = promedio(lambda a, b: (a ** 2.0 + b ** 2.0) / 2) - mu ** 2
    producto = promedio(lambda a, b: a * b * 1.0)
    correlacion = np.where(varianza > 1e-9, (producto - mu ** 2) / np.maximum(varianza, 1e-9), 1.0)

    # P(i, j) local: para cada par no ordenado {i, j}, fracción de pares de la ventana que valen {i, j}.
    # En la GLCM simétrica, si i ≠ j esa fracción se reparte en las casillas (i, j) y (j, i).
    asm = np.zeros(q.shape)                          # Σ P² (energía al cuadrado)
    entropia = np.zeros(q.shape)
    codigos = [np.where(v, np.minimum(a, b) * niveles + np.maximum(a, b), -1) for a, b, v, _ in pares]
    for k in np.unique(np.concatenate([c[c >= 0] for c in codigos])):
        i, j = divmod(int(k), niveles)
        frac = sum(_suma_rectangulo((c == k).astype(float), *rect)
                   for c, (_, _, _, rect) in zip(codigos, pares)) / cuenta
        if i == j:
            asm += frac ** 2
            entropia -= frac * np.log2(np.maximum(frac, 1e-300))
        else:
            asm += 2 * (frac / 2) ** 2
            entropia -= frac * np.log2(np.maximum(frac / 2, 1e-300))

    return {"contraste": contraste, "homogeneidad": homogeneidad, "energia": np.sqrt(asm),
            "correlacion": correlacion, "entropia": entropia}


def calcular_mapas(img, ventana=15, niveles=16):
    """Todos los mapas de MAPAS para una imagen y un tamaño de ventana.

    - intensidad: la imagen misma (no depende de la ventana);
    - media, varianza: promedio y varianza en la ventana;
    - cv: coeficiente de variación local σ/μ (la Parte 2 mostró que σ depende de μ;
      el cociente compensa en parte esa dependencia);
    - gradiente: magnitud de Sobel promediada en la ventana;
    - GLCM (contraste, homogeneidad, energía, correlación, entropía): mapas_glcm sobre
      la imagen cuantizada a `niveles` niveles.
    Devuelve dict {nombre: mapa}, todos float del tamaño de img.
    """
    img = img.astype(float)
    media, varianza = mapa_media_varianza(img, ventana)
    resultado = {
        "intensidad": img,
        "media": media,
        "varianza": varianza,
        "cv": np.sqrt(varianza) / np.maximum(media, 1.0),
        "gradiente": ndimage.uniform_filter(filters.sobel(img), ventana),
    }
    resultado.update(mapas_glcm(reducir_niveles(img, niveles), ventana, niveles))
    return resultado


def separabilidad(mapa, lesion, tejido):
    """Qué tan bien separa `mapa` los píxeles de `lesion` de los de `tejido`.

    Devuelve (auc, fisher):
    - auc: área bajo la curva ROC. Probabilidad de que un píxel de la lesión elegido al
      azar tenga un valor más alto que uno del tejido. 0,5 = no separa; 1 = la lesión
      siempre más alta; 0 = siempre más baja (separa igual de bien, al revés).
    - fisher: razón de Fisher F = (μ_lesión − μ_tejido)² / (σ²_lesión + σ²_tejido).
    """
    a, b = mapa[lesion], mapa[tejido]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    etiquetas = np.r_[np.ones(len(a)), np.zeros(len(b))]
    auc = roc_auc_score(etiquetas, np.r_[a, b])
    fisher = (a.mean() - b.mean()) ** 2 / max(a.var() + b.var(), 1e-12)
    return auc, fisher


def entradas(img):
    """Las dos entradas a comparar: la imagen limpia sin filtrar y la mediana 7×7 (Parte 3)."""
    return {"sin_filtrar": img.astype(float), "mediana_7x7": dominios.filtrar(img, "mediana", 2)}


def medir_caso(img, mascara, marcas, ventanas=VENTANAS, niveles=16):
    """Calcula todos los mapas (2 entradas × ventanas) en una imagen y mide su separabilidad.

    img: imagen limpia (artefactos.limpiar); mascara: lesión; marcas: píxeles reconstruidos
    por inpainting, que no se usan para medir.
    Las regiones son las de la Parte 3: interior de la lesión (sin la banda del borde) y
    anillo de tejido entre 5 y 20 px por fuera. Los mapas se calculan en un recorte
    alrededor de la lesión (con margen suficiente para la ventana más grande).

    Devuelve una lista de dicts: entrada, ventana (0 = sin ventana), mapa, auc, fisher,
    mediana del mapa en la lesión y en el tejido.
    """
    zona = dominios.recorte(mascara, margen=20 + max(ventanas) // 2 + 5)
    m, mk = mascara[zona], marcas[zona]
    lesion = m & ~dominios.contorno(m) & ~mk
    tejido = dominios.anillo(m) & ~mk

    filas = []
    for nombre_entrada, entrada in entradas(img).items():
        recorte = entrada[zona]
        for ventana in ventanas:
            for nombre_mapa, mapa in calcular_mapas(recorte, ventana, niveles).items():
                if nombre_mapa == "intensidad" and ventana != ventanas[0]:
                    continue                          # no depende de la ventana: una sola vez
                auc, fisher = separabilidad(mapa, lesion, tejido)
                filas.append({
                    "entrada": nombre_entrada,
                    "ventana": 0 if nombre_mapa == "intensidad" else ventana,
                    "mapa": nombre_mapa,
                    "auc": auc,
                    "fisher": fisher,
                    "mediana_lesion": np.median(mapa[lesion]) if lesion.any() else np.nan,
                    "mediana_tejido": np.median(mapa[tejido]) if tejido.any() else np.nan,
                })
    return filas
