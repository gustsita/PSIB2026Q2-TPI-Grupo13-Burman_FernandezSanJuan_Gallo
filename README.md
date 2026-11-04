# PSIB 2026Q2 – TPI – Grupo 13

**Caracterización y segmentación de tumores de mama en ecografía (dataset BUSI)**

Lucía Burman · Lucía Fernandez San Juan · Agustina María Gallo

El trabajo reconstruye las regiones tapadas por marcas del ecógrafo, estudia el ruido speckle de las ecografías mamarias, compara el filtrado en el dominio espacial y en el frecuencial según su efecto sobre la textura, construye mapas de características para decidir cómo segmentar la lesión y, con la lesión segmentada, analiza qué descriptores de forma, intensidad y textura diferencian lesiones benignas de malignas.

## Estructura

```
config.yaml            # DATA_DIR y OUTPUT_DIR: rutas al dataset y a los resultados
requirements.txt       # dependencias
scripts/check_env.py   # verifica dependencias y que el dataset esté en DATA_DIR
src/busi/
    config.py          # lee config.yaml
    datos.py           # carga de imágenes y máscaras de BUSI
    artefactos.py      # Parte 1: detección de marcas del ecógrafo + inpainting
    ruido.py           # Parte 2: caracterización del speckle
    dominios.py        # Parte 3: filtrado espacial vs. frecuencial y efecto en la textura
    mapas.py           # Parte 4: mapas de características (GLCM local, varianza, gradiente)
    segmentacion.py    # Parte 5: region growing, k-means, Chan-Vese, Dice/IoU
    descriptores.py    # Parte 6: descriptores BI-RADS + estadística benigno/maligno
notebooks/             # un notebook por parte (00 = exploración del dataset)
data/raw/              # dataset (NO se sube a GitHub)
outputs/               # figuras y tablas generadas (no se sube)
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

## Uso

Los notebooks importan el paquete desde `src/`:

```python
from busi.datos import listar_casos, cargar_caso

casos = listar_casos()                 # DataFrame: id, clase, ruta, máscaras
img, mascara = cargar_caso(casos.iloc[0])
```

## Plan de trabajo por módulos

Cada parte es un módulo en `src/busi/` (las funciones) y un notebook (las pruebas,
figuras y conclusiones). Los módulos traen la explicación del método y funciones con
`TODO` para completar. Se avanza de a una función por vez, con un commit cada una.

| Parte | Módulo | Notebook | Necesita | Estado | Resultado principal |
|---|---|---|---|---|---|
| 0 | `datos.py` | `00_exploracion` | dataset | terminada | Lesión hipoecoica; malignas con menos contraste; duplicados y máscaras múltiples |
| 1 | `artefactos.py` | `01_artefactos` | Parte 0 | terminada | Top-hat + inpainting biharmónico (PSNR 33,3 dB) |
| 2 | `ruido.py` | `02_ruido` | Parte 1 | terminada | Speckle Gamma/Nakagami, no Rayleigh (μ/σ ≈ 4,6); compresión log del equipo |
| 3 | `dominios.py` | `03_dominios` | Parte 0 | terminada | Textura sin filtrar; brillo y bordes sobre la mediana 7 × 7 |
| 4 | `mapas.py` | `04_mapas` | Parte 3 | terminada | Media local (AUC 0,88), σ/μ y gradiente |
| 5 | `segmentacion.py` | `05_segmentacion` | Parte 4 | terminada | Crecimiento de regiones, ventana 200 px: Dice 0,65 (453 imágenes de evaluación) |
| 6 | `descriptores.py` | `06_descriptores` | Partes 5 y máscaras | terminada | 22/26 descriptores significativos; clasificador AUC 0,99 con máscara manual (sesgada), 0,85 con segmentación propia |

El resumen de todo el trabajo, con los números y las decisiones de cada parte, está en
[`docs/RESUMEN.md`](docs/RESUMEN.md).

Las Partes 1, 2 y 3 se pueden hacer en paralelo, una por integrante. Las Partes 4 y 6
también pueden avanzar en paralelo con la 5, porque usan las máscaras manuales.

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
