# Resumen del TPI – Grupo 13

**Caracterización del speckle y segmentación de tumores de mama en ecografía (BUSI)**

Lucía Burman · Lucía Fernandez San Juan · Agustina María Gallo — PSIB 2026Q2

Este documento resume qué se hizo en cada parte, los resultados principales (con números) y las
decisiones que pasaron de una parte a la siguiente. El detalle, las figuras y las limitaciones
están en cada notebook (`notebooks/00` a `06`).

## Datos

BUSI (Al-Dhabyani et al., 2020): 780 ecografías de mama (437 benignas, 210 malignas, 133 normales),
equipo LOGIQ E9, con máscara manual de la lesión. Las partes 3 a 6 trabajan con las **647 imágenes
con lesión**.

## Cadena de decisiones

```
Parte 0 → lesión hipoecoica, poco contraste en malignas, duplicados, máscaras múltiples
Parte 1 → imágenes limpias de marcas: artefactos.limpiar() (top-hat + inpainting biharmónico)
Parte 2 → speckle Gamma/Nakagami (no Rayleigh), imagen ya comprimida: no aplicar log
           y el tejido de las malignas es distinto lejos de la lesión → descriptores relativos
Parte 3 → textura sin filtrar; brillo y bordes sobre la mediana 7×7
Parte 4 → mapas: media local (mediana 7×7, 15 px), σ/μ (31 px), gradiente (31 px)
Parte 5 → crecimiento de regiones sobre la mediana 7×7, ventana 200 px → Dice 0,65
Parte 6 → descriptores; la forma de la máscara manual está sesgada por el dibujo;
           sistema completo AUC 0,85
```

## Parte 0 – Exploración (`00_exploracion`)

- Tamaños variables (310–719 × 190–1048 px), PNG de 8 bits; 16 imágenes con marcas de color.
- Las malignas ocupan más de la imagen (mediana 12 % contra 4 %).
- La lesión es **hipoecoica** pero su histograma se superpone con el del tejido. Contraste
  lesión–tejido menor en malignas (mediana 44 contra 57 niveles de gris).
- 17 imágenes con más de una máscara; BUSI tiene imágenes **casi duplicadas**.

## Parte 1 – Marcas del ecógrafo (`01_artefactos`, `artefactos.py`)

- Detección: *top-hat* blanco (disco de radio 3), umbral 100, descarte de fibras horizontales por
  forma, dilatación de 2 px. Las marcas ocupan una mediana de 0,10 % de cada imagen.
- Reconstrucción: **inpainting biharmónico**, el mejor en la validación con 600 cruces sintéticas
  (PSNR mediano 33,3 dB contra 30,2 de Navier-Stokes y 29,5 de Telea; Wilcoxon p < 10⁻³⁰).
- Detecta el 96–100 % de las cruces sobre tejido con brillo ≤ 100, casi ninguna sobre tejido
  > 120 (limitación principal). El relleno conserva el 73 % de la variación local del tejido.

## Parte 2 – Ruido speckle (`02_ruido`, `ruido.py`)

3900 parches de tejido (5 por imagen) y 3,9 millones de píxeles.

- **No es Rayleigh:** Rayleigh gana por AIC en el 0,1 % de los parches; **Gamma** en el 62 % y
  **Nakagami** en el 38 % ($m$ mediano ≈ 5).
- μ/σ mediano **4,6** (no 1,91) y crece con el brillo. La varianza crece con la media con
  β ≈ 1,2 (no 2); después de aplicar log, β ≈ −0,8 (sobrecorrige).
- Causa principal: la **compresión logarítmica** del equipo (σ ≈ 23 constante en el tejido claro,
  D ≈ 35–45). Elegir D para que μ/σ dé 1,91 es un ajuste (siempre existe ese D); la evidencia es que,
  aun así, la **forma** del histograma sigue sin ser Rayleigh.
- El tejido de las **malignas** tiene μ/σ más alto (5,03 contra 4,46, p < 0,001) con el mismo brillo,
  lejos de la lesión → usar descriptores relativos en la Parte 6.
- Limitación: descartar parches con píxeles ≥ 250 deja poco representado el tejido muy claro.

## Parte 3 – Dominio espacial vs. frecuencial (`03_dominios`, `dominios.py`)

Seis filtros (media, gaussiano, mediana; ideal, Butterworth, gaussiano frecuencial) con suavizado
emparejado, en las 647 imágenes.

- Para filtros lineales el dominio no importa (el gaussiano da lo mismo en los dos); importa la
  forma de $H$. El **ideal** es el peor (*ringing*).
- La **mediana** preserva mejor el borde (contraste de borde 1,75 contra 1,61–1,64 de los lineales).
- Los filtros sacan poco ruido (ENL × 2 como máximo) y con σ ≥ 2 **borran la diferencia de textura**
  lesión–tejido.
- Decisión: textura **sin filtrar**; brillo y bordes sobre la **mediana 7 × 7**.

## Parte 4 – Mapas de características (`04_mapas`, `mapas.py`)

10 mapas × 3 ventanas × 2 entradas, separabilidad lesión vs. anillo de tejido (AUC dentro de cada imagen).

- El **brillo** separa mejor: media local AUC 0,88; ningún mapa de textura supera a la imagen
  (mejor: coeficiente de variación, 0,78). La hipótesis de textura se cumple solo en parte.
- Todos los mapas separan peor en las malignas (media local 0,80 contra 0,92).
- Elegidos (no redundantes): **media local** (mediana 7 × 7, 15 px), **σ/μ** (sin filtrar, 31 px) y
  **gradiente** (sin filtrar, 31 px).

## Parte 5 – Segmentación (`05_segmentacion`, `segmentacion.py`)

Semiautomática: un clic simulado (el punto más interior de la máscara), seis métodos, parámetros
elegidos en 194 imágenes y evaluados en las 453 restantes contra la lesión de la semilla.

| Método | Dice medio | IoU medio |
|---|---|---|
| **Crecimiento de regiones sobre la mediana 7 × 7** (tolerancia 30, ventana 200 px) | **0,654** | **0,513** |
| Crecimiento de regiones sobre la media local | 0,638 | 0,491 |
| Chan-Vese (media local) | 0,601 | 0,466 |
| k-means (media local) | 0,593 | 0,455 |
| Otsu (media local) | 0,582 | 0,445 |
| k-means (media, σ/μ, gradiente) | 0,568 | 0,432 |

- El mejor le gana a los otros cinco (Wilcoxon pareado, p ≤ 10⁻⁶). Casi no hubo sobreajuste
  (0,669 en ajuste, 0,654 en evaluación).
- La **ventana de trabajo fija** (200 px alrededor de la semilla) es el parámetro más importante:
  sin ella todos los métodos caen a 0,40–0,56.
- Malignas peor segmentadas (0,61 contra 0,67, p = 10⁻⁶).
- **¿La textura suma?** Solo en las malignas (k-means: 0,535 → 0,579, p = 2·10⁻⁵); en las benignas
  empeora; en total, empate.
- Segmentar sobre la **imagen** es mejor que sobre la media local: la media local corre el borde,
  algo que el AUC de la Parte 4 no medía.
- Las ventanas de 31 px **deforman las lesiones chicas** (k-means con 3 mapas: 0,47 contra 0,65).
- Fallos: semillas sobre ecos internos claros, fugas a tejido oscuro vecino, lesiones grandes
  cortadas por la ventana.

## Parte 6 – Descriptores y benigno vs. maligno (`06_descriptores`, `descriptores.py`)

26 descriptores (forma, intensidad, textura GLCM con cuantización fija y textura relativa al anillo).

- **Una lesión por imagen** (la de la semilla); en las 15 imágenes con varias lesiones se pierde
  ~29 % del área marcada.
- **Duplicados:** huellas de 64 × 64 sin tendencia, correlación ≥ 0,4 (umbral elegido porque desde ahí
  el 93–95 % de los pares es de la misma clase, contra 41 % por azar). 264 de las 647 imágenes tienen
  una casi-gemela (498 escenas). 7 grupos mezclan clases (por ejemplo `benign (433)` y
  `malignant (145)` son la misma imagen).
- **Mann-Whitney + Holm:** 22 de 26 descriptores difieren. Los más fuertes: solidez, circularidad,
  correlación GLCM, área y **sombra posterior** (malignas 0,84 = sombra; benignas 1,22 = refuerzo).
  Con una imagen por grupo de duplicados siguen 20 de 22.
- **Sesgo del dibujo:** las máscaras benignas son casi elipses (IoU con su elipse 0,93 contra 0,82;
  esa medida sola da AUC 0,94). La forma calculada sobre la máscara manual mezcla forma de la lesión
  y estilo de quien dibujó, que conocía el diagnóstico.
- **Textura:** la del anillo de tejido separa tanto como la de la lesión (confirma la Parte 2).
  Relativa al entorno, la lesión maligna es **menos homogénea** (AUC 0,64): la hipótesis de
  "textura más heterogénea" se cumple solo en términos relativos.
- **Clasificador** (regresión logística, `class_weight="balanced"`, `StratifiedGroupKFold` por grupos
  de duplicados):

| Descriptores | Máscara manual (AUC) | Segmentación propia (AUC) |
|---|---|---|
| Forma | 0,993 | 0,748 |
| Intensidad | 0,755 | 0,780 |
| Textura (absoluta + relativa) | 0,867 | 0,822 |
| **Todos** | 0,988 (sens. 0,95, esp. 0,98) | **0,853 (sens. 0,82, esp. 0,76)** |

- Con nuestra segmentación se pierden la solidez y la circularidad, pero se conservan la intensidad,
  la textura, el área, el alto/ancho y la excentricidad. Los duplicados casi no inflan el AUC (≤ 0,006).
- **Estimación realista del sistema completo** (clic → segmentación → descriptores → clasificador):
  **AUC ≈ 0,85**.

## Conclusiones generales

1. Las ecografías de BUSI no son la amplitud del eco sino una imagen comprimida y procesada: el
   speckle no es Rayleigh y no conviene aplicar log antes de filtrar.
2. El speckle es ruido para la segmentación pero información para la textura: por eso la textura se
   mide sin filtrar y el brillo con la mediana.
3. Lo que más separa la lesión del tejido es el **brillo**; la textura ayuda solo donde el brillo no
   alcanza (malignas).
4. Un método clásico, de un clic, llega a Dice ≈ 0,65; la ventana de trabajo y el lugar del clic pesan
   más que la elección del método.
5. Las máscaras manuales de BUSI llevan información del diagnóstico en la forma del contorno: los
   resultados casi perfectos con ellas no se trasladan a un sistema automático.
