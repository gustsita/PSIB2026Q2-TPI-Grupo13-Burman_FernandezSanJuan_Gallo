# PSIB 2026Q2 – TPI – Grupo 13

**Caracterización y segmentación de tumores de mama en ecografía (dataset BUSI)**

Lucía Burman · Lucía Fernandez San Juan · Agustina María Gallo

El trabajo reconstruye las regiones tapadas por marcas del ecógrafo, estudia el ruido speckle de las ecografías mamarias, compara el filtrado en el dominio espacial y en el frecuencial según su efecto sobre la textura, construye mapas de características para decidir cómo segmentar la lesión y, con la lesión segmentada, analiza qué descriptores de forma, intensidad y textura diferencian lesiones benignas de malignas.

El sistema completo es semiautomático: a partir de una ecografía y un clic dentro de la lesión llega a Dice 0,65 en la segmentación y AUC 0,85 en la clasificación benigna/maligna. Los resultados y las decisiones de cada parte están resumidos en [`docs/RESUMEN.md`](docs/RESUMEN.md).

## Estructura

```
config.yaml            # DATA_DIR y OUTPUT_DIR: rutas al dataset y a los resultados
interfaz.py            # interfaz gráfica: cargar una ecografía, hacer clic en la lesión y ver cada paso
requirements.txt       # dependencias
scripts/
    check_env.py       # verifica dependencias y que el dataset esté en DATA_DIR
    entrenar_modelo.py # entrena y evalúa el clasificador final, y lo guarda en modelos/
    predecir.py        # clasifica ecografías nuevas (casos incógnita)
src/busi/
    config.py          # lee config.yaml
    datos.py           # carga de imágenes y máscaras de BUSI
    artefactos.py      # Parte 1: detección de marcas del ecógrafo + inpainting
    ruido.py           # Parte 2: caracterización del speckle
    dominios.py        # Parte 3: filtrado espacial vs. frecuencial y efecto en la textura
    mapas.py           # Parte 4: mapas de características (GLCM local, varianza, gradiente)
    segmentacion.py    # Parte 5: region growing, k-means, Chan-Vese, Dice/IoU
    descriptores.py    # Parte 6: descriptores BI-RADS + estadística benigno/maligno
    modelo.py          # camino completo de un caso nuevo: limpiar, segmentar, describir, clasificar
modelos/               # clasificador entrenado (clasificador_busi.joblib)
notebooks/             # un notebook por parte (00 = exploración del dataset)
data/raw/              # dataset (NO se sube a GitHub)
outputs/               # resultados generados por los notebooks y scripts (no se sube)
docs/                  # anteproyecto, material de la cátedra y RESUMEN.md
```

## Puesta en marcha

Requiere Python 3.10 o superior.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_env.py
```

### Windows (PowerShell)

Desde la raíz del proyecto (`PSIB_TPI_GRUPO_13`):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\check_env.py
```

Si Windows no reconoce `python`, usar `py` en su lugar. Si PowerShell bloquea la activación,
permitir scripts para la sesión actual y volver a intentarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Para Jupyter / VS Code, elegir el kernel `.venv` (o registrarlo con
`python -m ipykernel install --user --name psib-tpi-g13 --display-name "PSIB TPI Grupo 13"`).

## Datos

Dataset: **Breast Ultrasound Images Dataset (BUSI)**, Al-Dhabyani et al., *Data in Brief* 28 (2020) 104863.
Kaggle: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>

1. Descargar el zip desde Kaggle (botón **Download**; requiere cuenta) o por terminal:
   ```bash
   curl -L -o data/raw/busi.zip https://www.kaggle.com/api/v1/datasets/download/aryashah2k/breast-ultrasound-images-dataset
   ```
2. Descomprimir dentro de `data/raw/`:
   ```bash
   unzip data/raw/busi.zip -d data/raw/
   ```
   Tiene que quedar `data/raw/Dataset_BUSI_with_GT/{benign,malignant,normal}/`.
   Si la carpeta tiene otro nombre, ajustar `DATA_DIR` en `config.yaml`.
3. Correr `python scripts/check_env.py`: debería mostrar 437 benign, 210 malignant y 133 normal.

## Interfaz gráfica

Es el objetivo opcional del anteproyecto: cargar una ecografía, marcar la lesión con un clic y ver
los mapas, la segmentación y los descriptores. Desde la raíz del repo, con el entorno activado:

```bash
streamlit run interfaz.py
```

Se abre en el navegador (`http://localhost:8501`). En la barra de la izquierda se elige una imagen de
BUSI o se sube una propia; después se hace **un clic dentro de la lesión**. Aparecen la clasificación
(benigna/maligna y su probabilidad), la segmentación (y el Dice con la máscara del radiólogo si la
imagen es de BUSI) y tres pestañas con cada paso: las marcas del ecógrafo que se borraron (Parte 1),
los mapas de características (Parte 4) y los 26 descriptores comparados con las medianas de cada
clase (Parte 6). Usa el mismo modelo de `modelos/` que `scripts/predecir.py`.

### Versión en línea (sin instalar nada)

**Link:** _se completa al publicarla en Streamlit Community Cloud_

Como el dataset no se sube a GitHub, si no encuentra `data/raw/` la interfaz usa las 8 imágenes de
`ejemplos/` (4 benignas y 4 malignas de BUSI, con su máscara). Entre ellas hay dos en las que el
modelo se equivoca (`benign (342)` y `malignant (169)`), para mostrar también sus límites. Además se
puede subir cualquier ecografía propia.

Para publicarla: en [share.streamlit.io](https://share.streamlit.io), *Create app* → este repositorio,
rama `main`, archivo `interfaz.py` (en *Advanced settings*, Python 3.12). Instala lo de
`requirements.txt`; `scikit-learn` está fijado en la versión con la que se entrenó el modelo.

## Clasificar casos nuevos (casos incógnita)

El clasificador ya entrenado está en `modelos/`, así que para usarlo alcanza con instalar las dependencias: no hace falta el dataset. El método es semiautomático y necesita un punto dentro de la lesión (la semilla). Hay tres formas de darlo:

```bash
# Abre la imagen para hacer clic en la lesión
python scripts/predecir.py imagen.png

# Con la semilla escrita como fila,columna
python scripts/predecir.py imagen.png --semilla 210,305

# Todas las imágenes de una carpeta, tomando la semilla de <nombre>_mask*.png
python scripts/predecir.py carpeta/ --usar-mascaras
```

Si el nombre del archivo tiene espacios, va entre comillas. El resultado aparece en la terminal, por ejemplo `maligna  (P(maligna) = 0.77)`. En `outputs/casos_incognita/` quedan una figura por imagen, con las marcas detectadas, la segmentación y los índices de la lesión, y la tabla `resultados.csv` con los 26 descriptores.

**Limitaciones:** el resultado depende del punto elegido y el modelo asume que hay una lesión, así que no detecta imágenes normales. El rendimiento esperable es el de la validación cruzada: AUC 0,85, sensibilidad 0,82 y especificidad 0,76. Es una herramienta de apoyo, no un diagnóstico.

Para volver a entrenar el modelo (requiere el dataset, tarda unos minutos):

```bash
python scripts/entrenar_modelo.py
```

## Partes del trabajo

Cada parte tiene un módulo en `src/busi/` (las funciones) y un notebook (el análisis, las figuras y las conclusiones).

| Parte | Módulo | Notebook | Necesita | Estado | Resultado principal |
|---|---|---|---|---|---|
| 0 | `datos.py` | `00_exploracion` | dataset | terminada | Lesión hipoecoica; malignas con menos contraste; duplicados y máscaras múltiples |
| 1 | `artefactos.py` | `01_artefactos` | Parte 0 | terminada | Top-hat + inpainting biharmónico (PSNR 33,3 dB) |
| 2 | `ruido.py` | `02_ruido` | Parte 1 | terminada | Speckle Gamma/Nakagami, no Rayleigh (μ/σ ≈ 4,6); compresión log del equipo |
| 3 | `dominios.py` | `03_dominios` | Parte 0 | terminada | Textura sin filtrar; brillo y bordes sobre la mediana 7 × 7 |
| 4 | `mapas.py` | `04_mapas` | Parte 3 | terminada | Media local (AUC 0,88), σ/μ y gradiente |
| 5 | `segmentacion.py` | `05_segmentacion` | Parte 4 | terminada | Crecimiento de regiones, ventana 200 px: Dice 0,65 (453 imágenes de evaluación) |
| 6 | `descriptores.py` | `06_descriptores` | Partes 5 y máscaras | terminada | 22/26 descriptores significativos; clasificador AUC 0,99 con máscara manual (sesgada), 0,85 con segmentación propia |

## Reproducir los resultados

Los notebooks se corren en orden, desde `notebooks/`, con el kernel `.venv`. Los resultados de los cálculos largos se guardan en `outputs/`, que no se sube a GitHub: la primera vez se recalculan y después se leen de ahí. La Parte 6 usa las máscaras que genera la Parte 5, así que `05_segmentacion` va antes que `06_descriptores`.

Con el repo limpio, en una compu de 10 núcleos, los notebooks 02 a 06 tardaron en total unos 40 minutos:

| Notebook | Tiempo aproximado |
|---|---|
| `02_ruido` | 5 min |
| `03_dominios` | 4 min |
| `04_mapas` | 10 min |
| `05_segmentacion` | 17 min |
| `06_descriptores` | 3 min |

Para usar las funciones desde otro notebook:

```python
from busi.datos import listar_casos, cargar_caso

casos = listar_casos()                 # DataFrame: id, clase, ruta, máscaras
img, mascara = cargar_caso(casos.iloc[0])
```

## Flujo de trabajo en Git

1. Antes de empezar: `git switch main && git pull`.
2. Rama por tarea: `git switch -c parte2-ruido`.
3. Commits chicos y frecuentes, con mensajes que digan qué se hizo:
   `ruido: extraer parches homogéneos lejos de la lesión`,
   `ruido: ajuste Rayleigh/Gamma y test KS`.
4. `git push -u origin parte2-ruido` y abrir un Pull Request a `main`. Otra integrante
   lo revisa y lo mergea.
5. Cada integrante commitea con su propio usuario de GitHub: la cátedra audita la
   distribución de aportes entre integrantes.
