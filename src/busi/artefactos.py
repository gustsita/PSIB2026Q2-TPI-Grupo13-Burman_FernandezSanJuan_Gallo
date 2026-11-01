"""Parte 1 – Reconstrucción de las marcas del ecógrafo (calipers, textos, escalas).

No reduce el speckle: saca estructuras que no son tejido para que no contaminen
la caracterización del ruido, los mapas de textura ni la segmentación.

Pasos:
1. Detección: las marcas son pequeñas y mucho más brillantes que su entorno.
   Top-hat blanco (img − apertura(img)) + umbral alto + etiquetado, filtrando las
   componentes por tamaño y forma. Dilatar un poco la máscara para cubrir el halo.
2. Reconstrucción: inpainting de los píxeles detectados a partir de la vecindad
   (cv2.inpaint con INPAINT_TELEA o INPAINT_NS, o skimage.restoration.inpaint_biharmonic).
3. Validación: pegar marcas sintéticas sobre regiones limpias, reconstruir y comparar
   la región con la original (MSE, PSNR).
"""

import cv2
import numpy as np
from skimage import measure, morphology, restoration


def tophat(img, radio=3):
    """Top-hat blanco: img − apertura(img) con un disco de radio `radio`.

    La apertura borra las estructuras claras más finas que el disco; al restarla
    quedan solo esas estructuras. Las marcas miden 3 px de grosor, así que un disco
    de radio 3 (diámetro 7) las borra por completo, incluso donde son más anchas
    (centro de las cruces, números).
    """
    return morphology.white_tophat(img, morphology.disk(radio))


def es_fibra(region, elongacion_min=3, largo_min=8, angulo_max=35):
    """True si la componente parece una fibra del tejido y no una marca.

    Las fibras son alargadas y aproximadamente horizontales (el tejido mamario se
    dispone en capas paralelas a la piel). Las marcas son cruces, puntos y letras:
    compactas, o alargadas pero verticales (el "1", la "I").

    region: un elemento de skimage.measure.regionprops.
    angulo_max: máxima inclinación respecto de la horizontal, en grados.
    """
    largo = region.axis_major_length
    elongacion = largo / max(region.axis_minor_length, 1)
    # orientation de skimage es el ángulo entre el eje de las filas y el eje mayor:
    # vale ±90° para una componente horizontal y 0° para una vertical
    inclinacion = 90 - abs(np.degrees(region.orientation))
    return elongacion >= elongacion_min and largo >= largo_min and inclinacion <= angulo_max


def detectar_marcas(img, radio_tophat=3, umbral=100, dilatar=2, descartar_fibras=True):
    """Máscara booleana con las marcas del ecógrafo.

    1. Top-hat con disco de radio `radio_tophat` y umbral `umbral`: candidatos.
    2. Etiquetado de componentes conexas y descarte de las que parecen fibras.
    3. Dilatación de `dilatar` px para cubrir el borde suavizado de las marcas y los
       píxeles de sus extremos que quedan por debajo del umbral. El valor 2 se eligió con
       la validación de marcas sintéticas (sección 5.5 del notebook 01): con 1 px quedan
       píxeles de la marca sin cubrir y con 3 px el hueco crece y el relleno empeora.
    """
    candidatos = tophat(img, radio_tophat) >= umbral

    marcas = candidatos
    if descartar_fibras:
        etiquetas = measure.label(candidatos, connectivity=2)
        fibras = [r.label for r in measure.regionprops(etiquetas) if es_fibra(r)]
        marcas = candidatos & ~np.isin(etiquetas, fibras)

    if dilatar:
        marcas = morphology.dilation(marcas, morphology.disk(dilatar))
    return marcas


def reconstruir(img, mascara_marcas, radio=3, metodo="biharmonico"):
    """Rellena los píxeles de `mascara_marcas` a partir de su vecindad (inpainting).

    metodo:
    - "telea": avanza desde el borde del hueco hacia adentro; cada píxel nuevo es un
      promedio ponderado de los píxeles conocidos a menos de `radio` px.
    - "ns" (Navier-Stokes): prolonga hacia adentro del hueco las líneas de igual
      nivel de gris (isofotas), para continuar los bordes que llegan al hueco.
    - "biharmonico": rellena con la superficie más suave que coincide con el borde
      del hueco (resuelve la ecuación biharmónica). `radio` no se usa.

    Por defecto "biharmonico": en la validación con marcas sintéticas (sección 5 del
    notebook 01) fue el de menor error (PSNR mediano 33,3 dB vs 30,2 de NS y 29,5 de Telea).
    """
    if metodo == "biharmonico":
        rec = restoration.inpaint_biharmonic(img, mascara_marcas)   # devuelve float en [0, 1]
        return np.clip(rec * 255, 0, 255).round().astype(np.uint8)
    flag = cv2.INPAINT_TELEA if metodo == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img.astype(np.uint8), mascara_marcas.astype(np.uint8), radio, flag)


def limpiar(img):
    """Detecta las marcas del ecógrafo y las reconstruye con el método elegido.

    Devuelve (img_limpia, mascara_marcas). Es la función que usan las partes siguientes.
    """
    marcas = detectar_marcas(img)
    return reconstruir(img, marcas), marcas


def agregar_marcas_sinteticas(img, n=3, rng=None, prohibida=None, tam=13, grosor=3,
                              valor=220, media_min=20, max_intentos=500):
    """Copia de img con n cruces tipo caliper dibujadas, y la máscara de dónde se dibujaron.

    Sirve para validar: como se conoce la imagen original debajo de la marca, se puede
    medir qué tan bien se detecta y se reconstruye.

    Las cruces imitan el perfil medido en las reales (sección 1 del notebook): un núcleo de
    `grosor` = 3 px con `valor` = 220 (las reales van de ~210 a 233) y 1 px de transición con
    la mitad de opacidad (~145) de **un solo lado** del trazo, como en los perfiles reales
    (77, 145, 213, 233, 202, 81 y 66, 212, 221, 222, 144, 98).
    Se descartan posiciones:
    - a menos de tam px del borde de la imagen;
    - que tocan `prohibida` (por ejemplo, las marcas reales ya detectadas);
    - sobre fondo oscuro (media de la zona < media_min), donde no hay tejido;
    - que se superponen con otra cruz sintética.

    Devuelve (img_marcada, mascara_sintetica, centros), con centros = lista de (fila, col).
    """
    if rng is None:
        rng = np.random.default_rng()
    alto, ancho = img.shape
    ocupada = np.zeros(img.shape, bool) if prohibida is None else prohibida.copy()
    marcada = img.copy()
    mascara = np.zeros(img.shape, bool)
    centros = []
    medio = tam // 2 + 2                     # medio lado de la zona que ocupa una cruz

    for _ in range(max_intentos):
        if len(centros) == n:
            break
        y = int(rng.integers(tam, alto - tam))
        x = int(rng.integers(tam, ancho - tam))
        zona = (slice(y - medio, y + medio + 1), slice(x - medio, x + medio + 1))
        if ocupada[zona].any() or img[zona].mean() < media_min:
            continue

        # núcleo de la cruz: dos barras de `grosor` px de ancho y `tam` px de largo
        nucleo = np.zeros(img.shape, bool)
        g, l = grosor // 2, tam // 2
        nucleo[y - g:y + g + 1, x - l:x + l + 1] = True
        nucleo[y - l:y + l + 1, x - g:x + g + 1] = True
        # transición de 1 px de un solo lado: el núcleo corrido 1 px hacia abajo y a la derecha
        borde = np.roll(np.roll(nucleo, 1, axis=0), 1, axis=1) & ~nucleo

        alfa = nucleo + 0.5 * borde                   # opacidad: 1 en el núcleo, 0,5 en el borde
        marcada = np.round(alfa * valor + (1 - alfa) * marcada).astype(np.uint8)

        mascara |= alfa > 0
        ocupada[zona] = True
        centros.append((y, x))

    return marcada, mascara, centros


def mse_psnr(original, reconstruida, mascara):
    """MSE y PSNR calculados solo dentro de `mascara`."""
    dif = original.astype(float)[mascara] - reconstruida.astype(float)[mascara]
    mse = np.mean(dif ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf
    return mse, psnr
