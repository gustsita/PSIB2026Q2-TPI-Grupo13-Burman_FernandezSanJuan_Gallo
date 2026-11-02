"""Parte 3 – Filtrado en dominio espacial vs. frecuencial y efecto sobre la textura.

Pregunta guía: ¿qué le hace cada filtro a la textura de la imagen? No alcanza con
medir cuánto ruido sacó: hay que ver qué particularidades texturales conserva o
destruye cada dominio (bordes, granularidad del speckle, patrón interno de la lesión).

Pasos:
1. Filtros espaciales: media, gaussiano, mediana (varios tamaños de ventana).
2. Filtros frecuenciales: pasabajos ideal, Butterworth y gaussiano sobre la FFT
   (varias frecuencias de corte).
   Los parámetros se eligen **emparejados**: para cada nivel de suavizado σ (en px),
   cada filtro usa el parámetro "equivalente" a un gaussiano de desvío σ
   (ver parametros_equivalentes), así se comparan los dominios con el mismo suavizado.
3. Para cada imagen filtrada medir:
   - SNR / ENL (número equivalente de looks) en parches homogéneos,
   - preservación de bordes (gradiente medio sobre el contorno de la máscara),
   - descriptores GLCM dentro y fuera de la lesión,
   - espectro de potencia (qué frecuencias se van),
   - PSNR / SSIM respecto de la imagen sin filtrar (cuánto cambió).
4. Comparar dominios: tablas y figuras lado a lado.
"""

import cv2
import numpy as np
from scipy import fft as sfft
from scipy import ndimage
from skimage import filters, metrics, morphology
from skimage.feature import graycomatrix, graycoprops

from . import ruido


# ---- Niveles de suavizado emparejados -------------------------------------

SIGMAS = (1, 2, 3)          # niveles de suavizado: desvío σ (px) del gaussiano equivalente

# nombre del filtro -> dominio
FILTROS = {
    "media": "espacial",
    "gaussiano": "espacial",
    "mediana": "espacial",
    "ideal": "frecuencial",
    "butterworth": "frecuencial",
    "gaussiano_frec": "frecuencial",
}


def parametros_equivalentes(sigma):
    """Parámetros de cada filtro equivalentes a un gaussiano espacial de desvío `sigma` px.

    - Media y mediana: ventana L (impar) con la misma varianza que la gaussiana.
      Una ventana de L píxeles tiene varianza (L² − 1) / 12, así que L ≈ √(12σ² + 1).
      σ = 1, 2, 3  ->  L = 3, 7, 11.
    - Frecuenciales: frecuencia de corte D0 = 1 / (2πσ) en ciclos/píxel. La transformada
      de Fourier de una gaussiana de desvío σ es otra gaussiana de desvío 1 / (2πσ), así
      que el pasabajos gaussiano con ese D0 es el mismo filtro que el gaussiano espacial.
      El ideal y el Butterworth usan el mismo D0.
      σ = 1, 2, 3  ->  D0 ≈ 0,159; 0,080; 0,053 ciclos/píxel.
    """
    L = np.sqrt(12 * sigma ** 2 + 1)
    ventana = int(2 * round((L - 1) / 2) + 1)          # impar más cercano
    return {"ventana": ventana, "corte": 1 / (2 * np.pi * sigma)}


def filtrar(img, filtro, sigma):
    """Aplica `filtro` (una clave de FILTROS) con el nivel de suavizado `sigma`.

    Devuelve siempre una imagen float recortada a [0, 255], para que todas las métricas
    se calculen igual sea cual sea el filtro (el ideal, por ejemplo, puede dar valores
    fuera de rango por el ringing).
    """
    p = parametros_equivalentes(sigma)
    if filtro == "media":
        salida = filtro_media(img, p["ventana"])
    elif filtro == "gaussiano":
        salida = filtro_gaussiano(img, sigma)
    elif filtro == "mediana":
        salida = filtro_mediana(img, p["ventana"])
    elif filtro == "ideal":
        salida = pasabajos(img, p["corte"], "ideal")
    elif filtro == "butterworth":
        salida = pasabajos(img, p["corte"], "butterworth")
    elif filtro == "gaussiano_frec":
        salida = pasabajos(img, p["corte"], "gaussiano")
    else:
        raise ValueError(f"Filtro desconocido: {filtro}")
    return np.clip(salida.astype(float), 0, 255)


# ---- Dominio espacial ------------------------------------------------------

def filtro_media(img, tam=5):
    return ndimage.uniform_filter(img.astype(float), size=tam)


def filtro_gaussiano(img, sigma=1.5):
    return filters.gaussian(img.astype(float), sigma=sigma, preserve_range=True)


def filtro_mediana(img, tam=5):
    """Mediana en una ventana tam×tam.

    Usa cv2.medianBlur (necesita uint8): da el mismo resultado que
    scipy.ndimage.median_filter salvo en los bordes de la imagen, y es unas 30 veces
    más rápida con ventanas grandes.
    """
    return cv2.medianBlur(np.clip(img, 0, 255).astype(np.uint8), tam).astype(float)


# ---- Dominio frecuencial ---------------------------------------------------

def espectro(img):
    """FFT centrada de la imagen (np.fft.fftshift(np.fft.fft2(img)))."""
    return np.fft.fftshift(np.fft.fft2(img.astype(float)))


def distancia_frecuencia(shape):
    """D(u, v): distancia de cada punto del espectro centrado al centro, en ciclos/píxel.

    Va de 0 (el centro, la media de la imagen) a 0,5 en los bordes (la frecuencia más
    alta que se puede representar: un píxel claro, uno oscuro). Usar ciclos/píxel y no
    "píxeles de frecuencia" hace que un mismo corte signifique lo mismo en imágenes de
    distinto tamaño (BUSI tiene imágenes de 190 a 1048 px de ancho).
    """
    fu = np.fft.fftshift(np.fft.fftfreq(shape[0]))      # frecuencias verticales
    fv = np.fft.fftshift(np.fft.fftfreq(shape[1]))      # frecuencias horizontales
    return np.sqrt(fu[:, None] ** 2 + fv[None, :] ** 2)


def mascara_pasabajos(shape, corte, tipo="butterworth", orden=2):
    """Función de transferencia H(u, v) centrada.

    tipo: "ideal", "butterworth" o "gaussiano". `corte` (D0) en ciclos/píxel.
    - ideal:       H = 1 si D ≤ D0, 0 si no (corte abrupto)
    - butterworth: H = 1 / (1 + (D / D0)^(2·orden))   (transición suave; vale 0,5 en D0)
    - gaussiano:   H = exp(−D² / (2·D0²))              (vale 0,61 en D0)
    """
    D = distancia_frecuencia(shape)
    if tipo == "ideal":
        return (D <= corte).astype(float)
    if tipo == "butterworth":
        return 1 / (1 + (D / corte) ** (2 * orden))
    if tipo == "gaussiano":
        return np.exp(-D ** 2 / (2 * corte ** 2))
    raise ValueError(f"Tipo de pasabajos desconocido: {tipo}")


def filtrar_frecuencia(img, H):
    """Multiplica el espectro por H y vuelve al dominio espacial (parte real)."""
    G = espectro(img) * H
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))


def pasabajos(img, corte, tipo="butterworth", orden=2):
    """Filtra `img` con un pasabajos en frecuencia, con padding por reflexión.

    La FFT trata la imagen como si se repitiera periódicamente: sin padding, el filtro
    mezcla el borde de arriba con el de abajo (y el izquierdo con el derecho). Para
    evitarlo se agranda la imagen reflejándola en los bordes hasta ~el doble de tamaño,
    se filtra y se recorta la parte central. Se usa scipy.fft (la misma FFT que numpy,
    más rápida) con un tamaño "rápido" (next_fast_len).
    """
    alto, ancho = img.shape
    alto_p, ancho_p = sfft.next_fast_len(2 * alto), sfft.next_fast_len(2 * ancho)
    arriba, izquierda = (alto_p - alto) // 2, (ancho_p - ancho) // 2
    ampliada = np.pad(img.astype(float),
                      ((arriba, alto_p - alto - arriba), (izquierda, ancho_p - ancho - izquierda)),
                      mode="reflect")
    H = mascara_pasabajos(ampliada.shape, corte, tipo, orden)
    G = sfft.fftshift(sfft.fft2(ampliada, workers=-1)) * H
    filtrada = np.real(sfft.ifft2(sfft.ifftshift(G), workers=-1))
    return filtrada[arriba:arriba + alto, izquierda:izquierda + ancho]


def psd_radial(img, paso=0.01):
    """Densidad espectral de potencia promediada en anillos de igual frecuencia.

    PSD = |F(u, v)|², sin la media de la imagen (que solo aporta al centro). Se promedia
    en anillos de ancho `paso` ciclos/píxel, de 0 a 0,5.

    Antes de la FFT se multiplica la imagen por una ventana de Hann (vale 1 en el centro
    y baja suavemente a 0 en los bordes). Sin ventana, la FFT "ve" un salto entre bordes
    opuestos de la imagen (la trata como periódica), y ese salto aporta potencia a todas
    las frecuencias (fuga espectral). En ecografía la potencia propia de las frecuencias
    altas es ~10⁵ veces menor que la de las bajas, así que la fuga la tapa por completo.

    Devuelve (frecuencias, perfil): el centro de cada anillo y la PSD media en él.
    """
    img = img.astype(float)
    ventana = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    F = sfft.fftshift(sfft.fft2((img - img.mean()) * ventana, workers=-1))
    psd = np.abs(F) ** 2 / (ventana ** 2).sum()
    D = distancia_frecuencia(img.shape)
    bordes = np.arange(0, 0.5 + paso / 2, paso)
    anillo = np.digitize(D, bordes) - 1                  # 0 .. n-1 dentro de [0, 0,5)
    validos = anillo < len(bordes) - 1
    suma = np.bincount(anillo[validos], weights=psd[validos], minlength=len(bordes) - 1)
    cuenta = np.bincount(anillo[validos], minlength=len(bordes) - 1)
    return (bordes[:-1] + bordes[1:]) / 2, suma / np.maximum(cuenta, 1)


# ---- Regiones de la lesión -------------------------------------------------

def contorno(mascara, grosor=1):
    """Banda de 2·grosor + 1 px centrada en el borde de la máscara.

    Es mascara XOR erosión(mascara), pero con un poco de ancho hacia los dos lados:
    la máscara manual no sigue el borde con precisión de un píxel.
    """
    disco = morphology.disk(grosor)
    return (ndimage.binary_dilation(mascara, structure=disco)
            & ~ndimage.binary_erosion(mascara, structure=disco, border_value=1))


def anillo(mascara, desde=5, hasta=20):
    """Tejido que rodea la lesión: entre `desde` y `hasta` px por fuera del borde.

    Empieza a `desde` px para no incluir el borde mismo (y el halo de la lesión).
    """
    lejos = ndimage.binary_dilation(mascara, structure=morphology.disk(hasta))
    cerca = ndimage.binary_dilation(mascara, structure=morphology.disk(desde))
    return lejos & ~cerca


def recorte(mascara, margen=30):
    """Slices del rectángulo que contiene la lesión más `margen` px de cada lado."""
    filas, cols = np.nonzero(mascara)
    alto, ancho = mascara.shape
    return (slice(max(filas.min() - margen, 0), min(filas.max() + margen + 1, alto)),
            slice(max(cols.min() - margen, 0), min(cols.max() + margen + 1, ancho)))


# ---- Métricas de comparación -----------------------------------------------

def enl(parche):
    """Número equivalente de looks: (μ/σ)². Más alto = menos speckle."""
    parche = parche.astype(float)
    return (parche.mean() / parche.std()) ** 2


def preservacion_bordes(img_filtrada, mascara, excluir=None):
    """Gradiente medio sobre el contorno de la lesión.

    Contorno: banda de 3 px alrededor del borde de la máscara (ver contorno()).
    Gradiente: magnitud de Sobel (skimage.filters.sobel).
    `excluir`: píxeles que no se cuentan (por ejemplo, las marcas reconstruidas).
    """
    banda = contorno(mascara)
    if excluir is not None:
        banda &= ~excluir
    return filters.sobel(img_filtrada.astype(float))[banda].mean()


def glcm_region(img, region, niveles=32, distancias=(1, 4), props=("contrast", "homogeneity", "energy", "correlation")):
    """Propiedades de la GLCM calculadas solo con los píxeles de `region`.

    La imagen se cuantiza a `niveles` niveles de gris (de 0-255 a 1..niveles) y los
    píxeles de afuera de la región se ponen en 0. Se calcula la GLCM con niveles + 1
    niveles y se descarta la fila y la columna 0: así solo cuentan los pares de píxeles
    que caen los dos dentro de la región. Se promedian las 4 direcciones (0°, 45°, 90°,
    135°). La cuantización es fija (no depende de la imagen) para que los valores de
    distintos filtros sean comparables.

    Devuelve dict {f"{prop}_d{d}": valor}.
    """
    q = (np.clip(img, 0, 255) * niveles / 256).astype(np.uint8) + 1
    q[~region] = 0
    angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    P = graycomatrix(q, distancias, angulos, levels=niveles + 1, symmetric=True)[1:, 1:]
    resultado = {}
    for prop in props:
        valores = graycoprops(P, prop).mean(axis=1)     # promedio de las 4 direcciones
        for d, v in zip(distancias, valores):
            resultado[f"{prop}_d{d}"] = v
    return resultado


def medir_caso(img, mascara, marcas, rng, sigmas=SIGMAS, n_parches=5):
    """Filtra una imagen con todos los filtros y niveles y mide cada resultado.

    img: imagen ya limpia (artefactos.limpiar); mascara: lesión; marcas: píxeles
    reconstruidos por inpainting, que no se usan en ninguna medición.

    Devuelve (filas, perfiles):
    - filas: lista de dicts, una por filtro y nivel (más una fila "original" de
      referencia), con las métricas;
    - perfiles: dict {(filtro, sigma): PSD filtrada / PSD original, por frecuencia}.
    """
    img = img.astype(float)
    zona = recorte(mascara)
    m, mk = mascara[zona], marcas[zona]
    banda = contorno(m) & ~mk
    lesion = m & ~contorno(m) & ~mk
    tejido = anillo(m) & ~mk

    # parches homogéneos de tejido, lejos de la lesión y de las marcas (Parte 2)
    _, esquinas = ruido.extraer_parches(img, mascara | marcas, n=n_parches, rng=rng)
    tam = 32
    def parches_de(im):
        return [im[y:y + tam, x:x + tam] for y, x in esquinas]
    enl_original = [enl(p) for p in parches_de(img)]

    _, psd_original = psd_radial(img)
    grad_original = filters.sobel(img[zona])

    def medir(filtrada, filtro, sigma):
        grad = filters.sobel(filtrada[zona])
        fila = {
            "filtro": filtro,
            "dominio": FILTROS.get(filtro, "-"),
            "sigma": sigma,
            "ganancia_enl": np.median([enl(p) / e for p, e in zip(parches_de(filtrada), enl_original)])
                            if enl_original else np.nan,
            "borde_relativo": grad[banda].mean() / grad_original[banda].mean(),
            "contraste_borde": grad[banda].mean() / grad[tejido].mean(),
            "psnr": metrics.peak_signal_noise_ratio(img[zona], filtrada[zona], data_range=255)
                    if filtro != "original" else np.inf,
            "ssim": metrics.structural_similarity(img[zona], filtrada[zona], data_range=255),
        }
        for nombre, region in (("lesion", lesion), ("tejido", tejido)):
            for k, v in glcm_region(filtrada[zona], region).items():
                fila[f"{k}_{nombre}"] = v
        return fila

    filas = [medir(img, "original", 0)]
    perfiles = {}
    for filtro in FILTROS:
        for sigma in sigmas:
            filtrada = filtrar(img, filtro, sigma)
            filas.append(medir(filtrada, filtro, sigma))
            perfiles[(filtro, sigma)] = psd_radial(filtrada)[1] / np.maximum(psd_original, 1e-12)
    return filas, perfiles
