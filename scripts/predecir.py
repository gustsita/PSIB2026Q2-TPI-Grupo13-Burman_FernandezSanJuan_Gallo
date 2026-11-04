"""Clasifica ecografías nuevas (casos incógnita): benigna o maligna.

El método es semiautomático: necesita un punto dentro de la lesión. Hay tres formas de darlo:

    python scripts/predecir.py imagen.png                       # abre la imagen: hacer clic en la lesión
    python scripts/predecir.py imagen.png --semilla 210,305     # fila,columna
    python scripts/predecir.py carpeta/ --usar-mascaras         # toma la semilla de <nombre>_mask*.png

Si se pasa una carpeta se procesan todas sus imágenes. Guarda en outputs/casos_incognita/ una
figura por imagen con el contorno segmentado y la tabla resultados.csv.
El modelo asume que hay una lesión: no detecta imágenes normales.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from busi import modelo, segmentacion
from busi.config import load_config
from busi.datos import cargar_imagen, cargar_mascara

ES_MASCARA = re.compile(r"_mask(_\d+)?$")
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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


def guardar_figura(archivo, img, semilla, resultado):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8, 7))
    FigureCanvasAgg(fig)                 # sin pyplot: no interfiere con la ventana del clic
    ax = fig.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if resultado["mascara"].any():
        ax.contour(resultado["mascara"], levels=[0.5], colors="lime", linewidths=1.5)
    ax.plot(semilla[1], semilla[0], "r+", markersize=12)
    p = resultado["probabilidad_maligna"]
    ax.set_title("Sin segmentación válida" if p is None else f"{resultado['clase']} (P(maligna) = {p:.2f})")
    ax.axis("off")
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
        guardar_figura(salida / f"{ruta.stem}.png", resultado["imagen_limpia"], semilla, resultado)
        filas.append({"imagen": ruta.name, "fila": semilla[0], "columna": semilla[1],
                      "area_px": resultado["area_px"], "probabilidad_maligna": resultado["probabilidad_maligna"],
                      "clase": resultado["clase"]})
        print(f"{ruta.name}: {resultado['clase'] or 'sin segmentación válida'}"
              + ("" if resultado["clase"] is None else f"  (P(maligna) = {resultado['probabilidad_maligna']:.2f})"))

    tabla = salida / "resultados.csv"           # se agrega a lo anterior, para no perder otros casos
    pd.DataFrame(filas).to_csv(tabla, mode="a", header=not tabla.exists(), index=False)
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
