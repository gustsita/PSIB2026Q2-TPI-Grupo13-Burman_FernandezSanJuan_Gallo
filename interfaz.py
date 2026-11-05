"""Interfaz gráfica (objetivo opcional del anteproyecto).

Permite cargar una ecografía, marcar la lesión con un clic (la semilla de la segmentación) y ver
cada paso del método: las marcas del ecógrafo que se borraron (Parte 1), los mapas de
características (Parte 4), la segmentación (Parte 5) y los descriptores con la clasificación
benigna/maligna (Parte 6).

Uso, desde la raíz del repo y con el entorno activado:

    streamlit run interfaz.py

Se abre en el navegador. No reemplaza a los notebooks: es una forma de ver el método funcionando
sobre una imagen por vez.
"""

import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from PIL import Image, ImageDraw
from scipy import ndimage
from streamlit_image_coordinates import streamlit_image_coordinates

from busi import modelo, segmentacion
from busi.datos import cargar_imagen, cargar_mascara, listar_casos

ANCHO_CLIC = 620                 # px de pantalla de la imagen donde se hace clic
COLOR_SEGMENTACION = (0, 230, 0)
COLOR_MANUAL = (255, 215, 0)
COLOR_MARCAS = (255, 60, 30)

NOMBRES_DESCRIPTORES = {
    "area": "Área (px)", "alto_ancho": "Alto / ancho", "circularidad": "Circularidad",
    "solidez": "Solidez", "excentricidad": "Excentricidad", "irregularidad": "Irregularidad del contorno",
    "media": "Brillo medio", "desvio": "Desvío del brillo", "ecogenicidad": "Ecogenicidad relativa",
    "sombra": "Sombra posterior (< 1 = sombra)",
}


# ---- Carga (con caché: se calcula una sola vez) -------------------------------

@st.cache_resource
def cargar_modelo():
    return modelo.cargar()


@st.cache_data
def casos_busi():
    try:
        casos = listar_casos()
    except FileNotFoundError:
        return None
    return casos[casos["clase"] != "normal"].set_index("id")


@st.cache_data(show_spinner="Limpiando, segmentando y calculando descriptores...")
def analizar(img, semilla):
    """Corre el método completo sobre una imagen y una semilla (fila, columna)."""
    guardado = cargar_modelo()
    resultado = modelo.predecir(guardado, img, semilla)
    mapas = segmentacion.mapas_segmentacion(resultado["imagen_limpia"])
    return resultado, mapas


# ---- Dibujo -------------------------------------------------------------------

def a_rgb(img):
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8)] * 3)


def pintar(rgb, mascara, color, alfa=1.0):
    """Pinta los píxeles de `mascara` de `color` (alfa = opacidad)."""
    rgb = rgb.astype(float)
    rgb[mascara] = (1 - alfa) * rgb[mascara] + alfa * np.array(color)
    return rgb.astype(np.uint8)


def contorno(mascara, grosor=2):
    """Borde de la máscara, de `grosor` px, para dibujarlo sobre la imagen."""
    return mascara & ~ndimage.binary_erosion(mascara, iterations=grosor)


def con_semilla(rgb, semilla, radio=7):
    imagen = Image.fromarray(rgb)
    dibujo = ImageDraw.Draw(imagen)
    y, x = semilla
    dibujo.line([(x - radio, y), (x + radio, y)], fill=(255, 0, 0), width=3)
    dibujo.line([(x, y - radio), (x, y + radio)], fill=(255, 0, 0), width=3)
    return imagen


def mapa_en_color(mapa, mascara=None):
    """Mapa de características con la escala viridis (percentil 1 a 99) y el contorno de la lesión."""
    lim = np.percentile(mapa, [1, 99])
    normalizado = np.clip((mapa - lim[0]) / max(lim[1] - lim[0], 1e-12), 0, 1)
    rgb = (colormaps["viridis"](normalizado)[..., :3] * 255).astype(np.uint8)
    if mascara is not None and mascara.any():
        rgb = pintar(rgb, contorno(mascara), (255, 40, 40))
    return rgb


def recorte(mascara, forma, margen=60):
    filas, cols = np.nonzero(mascara)
    return (slice(max(filas.min() - margen, 0), min(filas.max() + margen + 1, forma[0])),
            slice(max(cols.min() - margen, 0), min(cols.max() + margen + 1, forma[1])))


# ---- Página -------------------------------------------------------------------

st.set_page_config(page_title="Ecografía de mama – Grupo 13", layout="wide")
st.title("Segmentación y caracterización de lesiones en ecografía de mama")
st.caption("PSIB 2026Q2 – TPI – Grupo 13 · Dataset BUSI · Herramienta de apoyo, **no** un diagnóstico.")

guardado = cargar_modelo()
casos = casos_busi()

# 1. Elegir la imagen
with st.sidebar:
    st.header("1. Imagen")
    opciones = ["Subir una imagen"] if casos is None else ["Imagen de BUSI", "Subir una imagen"]
    origen = st.radio("Origen", opciones, label_visibility="collapsed")
    mascara_manual = None
    if origen == "Imagen de BUSI":
        clase = st.selectbox("Clase según el dataset", ["benign", "malignant"],
                             format_func=lambda c: {"benign": "benignas", "malignant": "malignas"}[c])
        ids = sorted(casos.index[casos["clase"] == clase], key=lambda s: int(s.split("(")[1].rstrip(")")))
        id_caso = st.selectbox("Imagen", ids)
        img = cargar_imagen(casos.loc[id_caso, "ruta"])
        mascara_manual = segmentacion.lesion_principal(cargar_mascara(casos.loc[id_caso, "mascaras"], img.shape))
        nombre = id_caso
        mostrar_manual = st.checkbox("Mostrar la máscara del radiólogo (amarillo)", value=False)
    else:
        archivo = st.file_uploader("Ecografía (PNG o JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if archivo is None:
            st.info("Subí una imagen para empezar.")
            st.stop()
        img = np.array(Image.open(io.BytesIO(archivo.getvalue())).convert("L"))
        nombre = archivo.name
        mostrar_manual = False

    st.header("2. Semilla")
    st.write("Hacé **un clic dentro de la lesión** en la imagen de la izquierda.")
    if mascara_manual is not None and st.button("Usar el centro de la máscara (clic simulado)"):
        st.session_state["semilla"] = (nombre, tuple(int(v) for v in segmentacion.semilla_desde_mascara(mascara_manual)))

    m = guardado["metricas"]
    st.divider()
    st.markdown(f"**Rendimiento del sistema** (validación cruzada, {m['lesiones']} lesiones):  \n"
                f"AUC {m['auc']:.2f} · sensibilidad {m['sensibilidad']:.2f} · especificidad {m['especificidad']:.2f}")
    st.caption("El resultado depende del clic, y el modelo supone que hay una lesión: no detecta imágenes normales.")

# la semilla guardada solo vale para la imagen en la que se marcó
semilla = None
if st.session_state.get("semilla") and st.session_state["semilla"][0] == nombre:
    semilla = st.session_state["semilla"][1]

col_clic, col_resultado = st.columns([1.1, 1])

with col_clic:
    st.subheader(nombre)
    base = a_rgb(img)
    if mostrar_manual and mascara_manual is not None:
        base = pintar(base, contorno(mascara_manual), COLOR_MANUAL)
    imagen_clic = con_semilla(base, semilla) if semilla else Image.fromarray(base)
    ancho = min(ANCHO_CLIC, img.shape[1])
    clic = streamlit_image_coordinates(imagen_clic, width=ancho, key=f"clic_{nombre}")
    if clic is not None:
        escala = img.shape[1] / clic["width"]
        nueva = (int(round(clic["y"] * escala)), int(round(clic["x"] * escala)))
        if nueva != semilla:
            st.session_state["semilla"] = (nombre, nueva)
            st.rerun()
    st.caption("Cruz roja: semilla. Hacé clic de nuevo para moverla.")

if semilla is None:
    with col_resultado:
        st.info("Marcá la lesión con un clic para ver la segmentación y la clasificación.")
    st.stop()

resultado, mapas = analizar(img, semilla)
mascara = resultado["mascara"]

with col_resultado:
    st.subheader("Resultado")
    if resultado["clase"] is None:
        st.error("La segmentación quedó vacía. Probá con un clic más adentro de la lesión.")
        st.stop()
    p = resultado["probabilidad_maligna"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Clasificación", resultado["clase"].capitalize())
    c2.metric("P(maligna)", f"{p:.2f}")
    if mascara_manual is not None:
        c3.metric("Dice con el radiólogo", f"{segmentacion.dice(mascara, mascara_manual):.2f}")
    else:
        c3.metric("Área segmentada", f"{resultado['area_px']} px")
    st.progress(p, text="probabilidad de maligna (umbral 0,5)")
    if origen == "Imagen de BUSI":
        st.caption(f"Clase según el dataset: **{'maligna' if clase == 'malignant' else 'benigna'}**.")

    zona = recorte(mascara, img.shape)
    vista = a_rgb(resultado["imagen_limpia"])
    if mostrar_manual and mascara_manual is not None:
        vista = pintar(vista, contorno(mascara_manual), COLOR_MANUAL)
    vista = pintar(vista, contorno(mascara), COLOR_SEGMENTACION)
    st.image(con_semilla(vista, semilla).crop((zona[1].start, zona[0].start, zona[1].stop, zona[0].stop)),
             caption="Segmentación (verde)" + (" y máscara del radiólogo (amarillo)" if mostrar_manual else ""),
             width="stretch")

st.divider()
st.subheader("Cada paso del método")
tab_marcas, tab_mapas, tab_desc, tab_como = st.tabs(
    ["1. Marcas del ecógrafo", "2. Mapas de características", "3. Descriptores", "Cómo funciona"])

with tab_marcas:
    marcas = resultado["marcas"]
    a, b = st.columns(2)
    a.image(pintar(a_rgb(img), ndimage.binary_dilation(marcas), COLOR_MARCAS),
            caption=f"Original: marcas detectadas en rojo ({100 * marcas.mean():.2f} % de la imagen)",
            width="stretch")
    b.image(resultado["imagen_limpia"], caption="Limpia: las marcas se rellenaron por inpainting biharmónico",
            width="stretch", clamp=True)
    st.caption("Parte 1: top-hat (disco de radio 3) + umbral 100 + descarte de fibras + dilatación de 2 px. "
               "Las marcas sobre tejido muy claro pueden no detectarse.")

with tab_mapas:
    titulos = [("imagen", "Imagen (mediana 7×7): donde crece la segmentación"),
               ("media", "Media local (15 px): brillo"),
               ("cv", "Coeficiente de variación σ/μ (31 px)"),
               ("gradiente", "Gradiente promediado (31 px)")]
    columnas = st.columns(4)
    for col, (clave, titulo) in zip(columnas, titulos):
        col.image(mapa_en_color(mapas[clave][zona], mascara[zona]), caption=titulo, width="stretch")
    st.caption("Parte 4: los tres mapas que mejor separan la lesión del tejido. Contorno rojo: la segmentación. "
               "Amarillo = valores altos, violeta = bajos. La lesión suele ser oscura en la media local y "
               "tener σ/μ alto y poco gradiente.")

with tab_desc:
    valores, ref = resultado["descriptores"], guardado["referencia"]
    filas = []
    for clave, valor in valores.items():
        b, m_ = ref["benign"][clave], ref["malignant"][clave]
        filas.append({"Descriptor": NOMBRES_DESCRIPTORES.get(clave, clave), "Esta lesión": valor,
                      "Mediana benignas": b, "Mediana malignas": m_,
                      "Más parecido a": "benignas" if abs(valor - b) < abs(valor - m_) else "malignas"})
    tabla = pd.DataFrame(filas).set_index("Descriptor")
    st.dataframe(tabla.style.format({"Esta lesión": "{:.3f}", "Mediana benignas": "{:.3f}", "Mediana malignas": "{:.3f}"}),
                 height=600)
    st.caption("Parte 6: 26 descriptores de forma, intensidad y textura (GLCM con cuantización fija, en la lesión "
               "y relativa al anillo de tejido: `_rel`). Las medianas son de las lesiones de BUSI segmentadas con este "
               "mismo método. El clasificador (regresión logística) usa los 26 juntos: un descriptor solo no decide.")

with tab_como:
    st.markdown(f"""
1. **Limpieza** (Parte 1): se detectan las marcas del ecógrafo (*calipers*, textos) y se rellenan por *inpainting*.
2. **Mapas** (Parte 4): media local, σ/μ y gradiente. El brillo es lo que más separa la lesión del tejido.
3. **Segmentación** (Parte 5): crecimiento de regiones desde el clic sobre la imagen filtrada con la mediana 7×7,
   tolerancia 30 niveles de gris, dentro de una ventana de 200 px alrededor del clic, y refinamiento morfológico.
   En BUSI da Dice 0,65 en promedio.
4. **Descriptores y clasificación** (Parte 6): 26 descriptores y una regresión logística entrenada con las lesiones de
   BUSI segmentadas de la misma forma. AUC {m['auc']:.2f}, sensibilidad {m['sensibilidad']:.2f},
   especificidad {m['especificidad']:.2f}.

**Limitaciones:** el resultado cambia según dónde se haga clic; lesiones más grandes que la ventana de 200 px quedan
cortadas; el modelo supone que hay una lesión; se entrenó con un solo equipo y un solo hospital. Detalles en
`docs/RESUMEN.md` y en los notebooks.
""")
