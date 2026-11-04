"""Clasifica ecografías nuevas (casos incógnita): benigna o maligna.

El método es semiautomático: necesita un punto dentro de la lesión. Hay tres formas de darlo:

    python scripts/predecir.py imagen.png                       # abre la imagen: hacer clic en la lesión
    python scripts/predecir.py imagen.png --semilla 210,305     # fila,columna
    python scripts/predecir.py carpeta/ --usar-mascaras         # toma la semilla de <nombre>_mask*.png

Si se pasa una carpeta se procesan todas sus imágenes. Guarda en outputs/casos_incognita/ una
figura por imagen con el proceso (marcas detectadas, segmentación e índices de la lesión frente
a los valores típicos de cada clase) y la tabla resultados.csv con todos los descriptores.
El modelo asume que hay una lesión: no detecta imágenes normales.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from busi import modelo, segmentacion
from busi.config import load_config
from busi.datos import cargar_imagen, cargar_mascara

ES_MASCARA = re.compile(r"_mask(_\d+)?$")
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MARGEN_ZOOM = 50                 # px alrededor de la lesión en el panel de segmentación
FORMATO = "{:.2f}"
INDICES = [("Área (px)", "area"), ("Alto / ancho", "alto_ancho"), ("Circularidad", "circularidad"),
           ("Solidez", "solidez"), ("Irregularidad", "irregularidad"),
           ("Ecogenicidad", "ecogenicidad"), ("Sombra posterior", "sombra"),
           ("Homogeneidad relativa", "homogeneidad_d1_rel")]


def pedir_clic(img, nombre):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 7))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{nombre}: hacer clic dentro de la lesión")
    puntos = plt.ginput(1, timeout=0)
    plt.close()
    if not puntos:
        raise SystemExit("No se marcó ningún punto.")
    x, y = puntos[0]
    return int(round(y)), int(round(x))


def obtener_semilla(ruta, img, args):
    if args.semilla:
        return tuple(int(v) for v in args.semilla.split(","))
    if args.usar_mascaras:
        mascaras = sorted(ruta.parent.glob(f"{ruta.stem}_mask*{ruta.suffix}"))
        if not mascaras:
            raise SystemExit(f"{ruta.name} no tiene máscara.")
        return tuple(int(v) for v in segmentacion.semilla_desde_mascara(cargar_mascara(mascaras)))
    return pedir_clic(img, ruta.name)


def guardar_figura(archivo, nombre, img, semilla, resultado, guardado):
    """Figura con el proceso: marcas detectadas, segmentación e índices frente a los valores típicos."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(18, 6.5))
    FigureCanvasAgg(fig)                 # sin pyplot: no interfiere con la ventana del clic
    ax_marcas, ax_lesion, ax_tabla = fig.subplots(1, 3, width_ratios=[1, 1, 1.05])

    marcas = resultado["marcas"]
    ax_marcas.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_marcas.imshow(np.ma.masked_where(~marcas, marcas), cmap="autumn", alpha=0.9)
    ax_marcas.set_title(f"1. Original y marcas detectadas ({100 * marcas.mean():.2f} % de la imagen)")

    mascara = resultado["mascara"]
    ax_lesion.imshow(resultado["imagen_limpia"], cmap="gray", vmin=0, vmax=255)
    if mascara.any():
        ax_lesion.contour(mascara, levels=[0.5], colors="lime", linewidths=1.5)
        filas, columnas = np.flatnonzero(mascara.any(axis=1)), np.flatnonzero(mascara.any(axis=0))
        ax_lesion.set_xlim(columnas[0] - MARGEN_ZOOM, columnas[-1] + MARGEN_ZOOM)
        ax_lesion.set_ylim(filas[-1] + MARGEN_ZOOM, filas[0] - MARGEN_ZOOM)
    ax_lesion.plot(semilla[1], semilla[0], "r+", markersize=14)
    ax_lesion.set_title("2. Imagen limpia, clic (rojo) y segmentación (verde)")
    for ax in (ax_marcas, ax_lesion):
        ax.axis("off")

    ax_tabla.axis("off")
    p = resultado["probabilidad_maligna"]
    if p is None:
        fig.suptitle(f"{nombre}: sin segmentación válida", fontsize=16)
    else:
        fig.suptitle(f"{nombre}: {resultado['clase']}  (P(maligna) = {p:.2f})", fontsize=16, fontweight="bold")
        ref, valores = guardado["referencia"], resultado["descriptores"]
        celdas = [[etiqueta, FORMATO.format(valores[d]) if d != "area" else f"{valores[d]:.0f}",
                   FORMATO.format(ref["benign"][d]) if d != "area" else f"{ref['benign'][d]:.0f}",
                   FORMATO.format(ref["malignant"][d]) if d != "area" else f"{ref['malignant'][d]:.0f}"]
                  for etiqueta, d in INDICES]
        tabla = ax_tabla.table(cellText=celdas, colLabels=["Índice", "Este caso", "Benignas\n(mediana)", "Malignas\n(mediana)"],
                               colWidths=[0.46, 0.18, 0.18, 0.18], loc="upper center", cellLoc="center")
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(11)
        tabla.scale(1, 2.1)
        ax_tabla.set_title("3. Índices de la lesión")
        m = guardado["metricas"]
        ax_tabla.text(0.5, 0.02, f"Validación cruzada del sistema completo: AUC {m['auc']:.2f}, sensibilidad "
                      f"{m['sensibilidad']:.2f}, especificidad {m['especificidad']:.2f}.\n"
                      f"El modelo usa {len(guardado['columnas'])} descriptores; acá se muestran 8. Sombra < 1: hay sombra.\n"
                      "El resultado depende del clic. Apoyo al análisis, no un diagnóstico.",
                      ha="center", va="bottom", fontsize=9.5, transform=ax_tabla.transAxes)
    fig.tight_layout()
    fig.savefig(archivo, dpi=120, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Clasifica ecografías de mama: benigna o maligna.")
    parser.add_argument("ruta", type=Path, help="imagen o carpeta de imágenes")
    parser.add_argument("--semilla", help="punto dentro de la lesión, como fila,columna (solo para una imagen)")
    parser.add_argument("--usar-mascaras", action="store_true",
                        help="toma la semilla de <nombre>_mask*.png, junto a cada imagen")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.ruta.is_dir():
        if args.semilla:
            raise SystemExit("--semilla es para una sola imagen.")
        rutas = sorted(r for r in args.ruta.iterdir()
                       if r.suffix.lower() in EXTENSIONES and not ES_MASCARA.search(r.stem))
    else:
        rutas = [args.ruta]
    if not rutas:
        raise SystemExit(f"No hay imágenes en {args.ruta}.")

    guardado = modelo.cargar()
    salida = load_config()["OUTPUT_DIR"] / "casos_incognita"
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for ruta in rutas:
        img = cargar_imagen(ruta)
        semilla = obtener_semilla(ruta, img, args)
        if not (0 <= semilla[0] < img.shape[0] and 0 <= semilla[1] < img.shape[1]):
            raise SystemExit(f"La semilla {semilla} está fuera de la imagen {img.shape}.")
        resultado = modelo.predecir(guardado, img, semilla)
        guardar_figura(salida / f"{ruta.stem}.png", ruta.stem, img, semilla, resultado, guardado)
        filas.append({"imagen": ruta.name, "fila": semilla[0], "columna": semilla[1],
                      "area_px": resultado["area_px"], "probabilidad_maligna": resultado["probabilidad_maligna"],
                      "clase": resultado["clase"], **(resultado["descriptores"] or {})})
        print(f"{ruta.name}: {resultado['clase'] or 'sin segmentación válida'}"
              + ("" if resultado["clase"] is None else f"  (P(maligna) = {resultado['probabilidad_maligna']:.2f})"))

    tabla = salida / "resultados.csv"           # se agrega a lo anterior, para no perder otros casos
    anteriores = [pd.read_csv(tabla)] if tabla.exists() else []
    pd.concat(anteriores + [pd.DataFrame(filas)]).to_csv(tabla, index=False)
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
