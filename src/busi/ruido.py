"""Parte 2 – Caracterización del ruido speckle.

Idea: el speckle es ruido multiplicativo, g = f · n. En zonas de tejido homogéneo
f es casi constante, así que la distribución de g refleja la de n.

Pasos:
1. Elegir parches homogéneos de tejido (lejos de la lesión y de las marcas del ecógrafo).
2. Estudiar el histograma de cada parche y ajustar distribuciones candidatas
   (Rayleigh, Gamma, Nakagami); compararlas (log-verosimilitud / test KS).
3. Calcular la relación media/desvío (para speckle completamente desarrollado con
   amplitud Rayleigh, μ/σ ≈ 1,91) y ver si el dataset la cumple.
4. Verificar que log(g) = log(f) + log(n): el ruido pasa a ser aditivo y su varianza
   ya no depende del brillo local.

Ojo para la discusión: las imágenes del ecógrafo vienen con compresión logarítmica y
post-procesamiento, así que no hay que esperar que cumplan exactamente la teoría.
"""

import cv2
import numpy as np
from scipy import ndimage, stats

from .config import load_config


def extraer_parches(img, mascara, tam=32, n=10, margen=20, rng=None,
                    media_min=20, desvio_min=3, saturado=250, max_intentos=2000):
    """Devuelve n parches tam×tam de tejido, alejados al menos `margen` px de la lesión.

    Se sortean posiciones al azar y se descartan los parches que:
    - tocan la lesión dilatada `margen` px (disco de radio `margen`);
    - caen en el fondo negro fuera del sector de la sonda: media < media_min,
      desvío < desvio_min o algún píxel en 0;
    - tienen marcas del ecógrafo: algún píxel >= `saturado`;
    - se superponen con un parche ya elegido.

    Devuelve (parches, esquinas): un array (k, tam, tam) y las esquinas superiores
    izquierdas (fila, columna) de cada parche. k puede ser menor que n si la imagen
    no tiene suficiente tejido válido.
    """
    if rng is None:
        rng = np.random.default_rng(load_config()["SEED"])

    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margen + 1, 2 * margen + 1))
    prohibida = cv2.dilate(mascara.astype(np.uint8), disco).astype(bool)

    alto, ancho = img.shape
    parches, esquinas = [], []
    for _ in range(max_intentos):
        if len(parches) == n:
            break
        y = rng.integers(0, alto - tam + 1)
        x = rng.integers(0, ancho - tam + 1)
        if prohibida[y:y + tam, x:x + tam].any():
            continue

        parche = img[y:y + tam, x:x + tam]
        if parche.mean() < media_min or parche.std() < desvio_min or parche.min() == 0:
            continue
        if parche.max() >= saturado:
            continue

        parches.append(parche.copy())
        esquinas.append((y, x))
        prohibida[y:y + tam, x:x + tam] = True     # para no repetir ni superponer parches

    return np.array(parches).reshape(-1, tam, tam), np.array(esquinas).reshape(-1, 2)


def razon_media_desvio(parche):
    """μ/σ del parche. Comparar contra 1,91 (Rayleigh)."""
    parche = parche.astype(float)
    return parche.mean() / parche.std()


def ajustar_distribuciones(valores, candidatas=("rayleigh", "gamma", "nakagami"), loc_cero=True):
    """Ajusta cada distribución de scipy.stats a `valores` por máxima verosimilitud.

    Con loc_cero=True se fija loc = 0, como en el modelo físico (la amplitud del eco
    empieza en 0). Con loc_cero=False scipy también ajusta un corrimiento.

    Devuelve dict {nombre: {"params", "loglik", "aic", "ks", "p_ks"}}:
    - loglik: log-verosimilitud (más alta = mejor ajuste);
    - aic: 2·k − 2·loglik, penaliza los parámetros libres (Rayleigh tiene uno menos);
    - ks, p_ks: estadístico D y p-valor del test de Kolmogorov-Smirnov.
    """
    valores = np.asarray(valores, dtype=float).ravel()
    resultados = {}
    for nombre in candidatas:
        dist = getattr(stats, nombre)
        params = dist.fit(valores, floc=0) if loc_cero else dist.fit(valores)
        loglik = dist.logpdf(valores, *params).sum()
        k = len(params) - (1 if loc_cero else 0)       # loc fijo no cuenta como parámetro libre
        ks = stats.kstest(valores, nombre, args=params)
        resultados[nombre] = {
            "params": params,
            "loglik": loglik,
            "aic": 2 * k - 2 * loglik,
            "ks": ks.statistic,
            "p_ks": ks.pvalue,
        }
    return resultados


def varianza_vs_media(img, ventana=7):
    """Media y varianza local en cada píxel.

    Graficar varianza contra media: si el ruido es multiplicativo la varianza crece
    con la media; después de aplicar log, debería quedar aproximadamente constante.
    Usa var = E[g²] − E[g]², con los promedios locales calculados con un filtro
    promediador de ventana×ventana (scipy.ndimage.uniform_filter).

    Devuelve (media, varianza), dos arrays del mismo tamaño que img.
    """
    img = img.astype(float)
    media = ndimage.uniform_filter(img, size=ventana)
    media_cuadrados = ndimage.uniform_filter(img ** 2, size=ventana)
    varianza = np.clip(media_cuadrados - media ** 2, 0, None)   # evita negativos por redondeo
    return media, varianza
