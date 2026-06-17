# MPS Born Machine para detección de anomalías en NSL-KDD

Detector de intrusiones de red basado en una **Born Machine** implementada como
**Matrix Product State (MPS)**. El modelo se entrena **solo con tráfico normal**:
aprende la distribución de probabilidad del comportamiento benigno y, en
evaluación, una *log-verosimilitud negativa* (NLL) alta señala un evento poco
probable bajo ese modelo → probable ataque.

El repositorio cubre el ciclo completo: codificación de los datos, entrenamiento
con DMRG, evaluación como detector y un módulo de **explicabilidad** que
aprovecha la estructura tensorial del MPS (entropías, información mutua,
importancia de características, probabilidades condicionales/conjuntas, etc.).

---

## Idea en breve

- Cada conexión de red de NSL-KDD se discretiza en una secuencia de enteros, uno
  por característica (un "sitio" del MPS, con su dimensión física `d_k`).
- El MPS representa una función de onda `Ψ` sobre esas configuraciones; la
  probabilidad de una muestra es `P(v) = |Ψ(v)|² / Z` (de ahí "Born Machine").
- Se entrena maximizando la verosimilitud del tráfico normal mediante **DMRG**
  con actualizaciones de dos sitios y crecimiento adaptativo de la dimensión de
  enlace (bond dimension).
- En inferencia, `NLL(v) = -log P(v)` actúa como **score de anomalía**. El umbral
  se fija como un percentil de la NLL sobre tráfico **normal** retenido (nunca se
  usan ataques para calibrar el umbral).

---

## Estructura del repositorio

```
.
├── mps.py                     # Núcleo: clase MPS (amplitudes, normas, NLL, canonicalización, SVD, swaps...)
├── dmrg_trainer.py            # Entrenador DMRG (DMRGConfig + dmrg_train) con crecimiento de bond dim y early stopping
├── mps_explainability.py      # MPSExplainer: RDMs, entropías, información mutua, probabilidades marginales/condicionales
├── mps_generative.py          # MPSSampler: muestreo (incl. condicional) desde el MPS
│
├── encoder_nsl_kdd.py         # [Paso 1] Codifica NSL-KDD a tensores discretos + esquema
├── train_mps_nsl_kdd.py       # [Paso 2] Entrena el MPS solo con tráfico normal
├── evaluate_mps_nsl_kdd.py    # [Paso 3] Evalúa el MPS como detector (ROC/PR, umbrales, por familia...)
└── explain_mps_nsl_kdd.py     # [Paso 4] Calcula y grafica la explicabilidad del MPS entrenado
```

Componentes de **librería** (`mps.py`, `dmrg_trainer.py`, `mps_explainability.py`,
`mps_generative.py`) frente a **scripts ejecutables** del pipeline (los
`*_nsl_kdd.py`). `mps_generative.py` no lo usa el pipeline directamente, pero
forma parte de la librería para muestrear desde el modelo entrenado.

---

## Requisitos

- Python 3.9+
- `torch`
- `numpy`
- `pandas`
- `scikit-learn` (métricas en la evaluación)
- `matplotlib` (figuras de evaluación y explicabilidad)

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## Datos

Descarga el dataset **NSL-KDD** y coloca los splits en formato CSV (sin cabecera,
42 columnas) dentro del directorio de datos. Por defecto el pipeline usa
`./nsl_kdd`:

```
./nsl_kdd/
├── KDDTrain+.txt
└── KDDTest+.txt
```

Todas las etiquetas se mapean a familias (`normal`, `dos`, `probe`, `r2l`, `u2r`)
y a un indicador binario `is_attack` dentro del codificador.

---

## Uso del pipeline

Todos los scripts aceptan el directorio de datos como argumento. Si se omite, usan
`./nsl_kdd`.

```bash
# 1) Codificar NSL-KDD -> tensores + esquema
python encoder_nsl_kdd.py ./nsl_kdd

# 2) Entrenar el MPS (solo tráfico normal)
python train_mps_nsl_kdd.py ./nsl_kdd

# 3) Evaluar como detector de anomalías
python evaluate_mps_nsl_kdd.py ./nsl_kdd

# 4) Explicabilidad (tablas + figuras)
python explain_mps_nsl_kdd.py ./nsl_kdd
```

Cada paso lee los artefactos del anterior desde el mismo directorio, así que el
orden importa.

---

## Qué genera cada paso

Todo se escribe dentro del directorio de datos (`./nsl_kdd`).

### Paso 1 — `encoder_nsl_kdd.py`

```
./nsl_kdd/
├── train_X.pt              # LongTensor (n_train, n_features) codificado
├── test_X.pt               # LongTensor (n_test,  n_features) codificado
├── train_meta.pt           # is_attack, family_code, family_names, difficulty, label
├── test_meta.pt            # idem para test
└── encoding_schema.json    # physical_dims y spec por característica (vocab/edges/normal_value)
```

### Paso 2 — `train_mps_nsl_kdd.py`

```
./nsl_kdd/
├── mps_trained.pt          # MPS entrenado (formato MPS.save: config + tensores de sitio)
├── train_history.json      # historial por loop: train_nll, val_nll, lr, bond_dims, ...
└── train_log.jsonl         # log JSONL escrito durante el entrenamiento (un registro por loop)
```

### Paso 3 — `evaluate_mps_nsl_kdd.py`

```
./nsl_kdd/
├── eval_report.json        # informe legible por máquina (métricas + rutas de figuras/tablas)
│
├── evaluate_graphs/
│   ├── eval_nll_histograms.png     # histogramas de NLL normal vs ataque
│   ├── eval_roc_pr.png             # curvas ROC y Precision-Recall
│   ├── eval_threshold_sweep.png    # barrido del umbral (operating points)
│   └── eval_confusion.png          # matriz de confusión en el umbral elegido
│
└── evaluate_tables/
    ├── global_metrics.csv          # AUROC, AUPRC, etc.
    ├── metrics_per_threshold.csv   # métricas en cada percentil de umbral
    ├── per_family.csv              # rendimiento por familia de ataque
    ├── per_difficulty.csv          # rendimiento por nivel de dificultad
    ├── distribution_shift.csv      # comparación de distribuciones train/test
    ├── realised_fpr.csv            # FPR real obtenido vs el objetivo
    ├── threshold_sweep.csv         # datos del barrido de umbral
    └── test_scores.csv             # NLL por muestra de test + etiquetas
```

> El umbral se calibra como percentil de la NLL sobre el **split de validación
> normal** que el entrenador retuvo, reusando sus mismas funciones de partición
> para que no haya fuga de información de ataques ni *drift* respecto al
> entrenamiento.

### Paso 4 — `explain_mps_nsl_kdd.py`

```
./nsl_kdd/
├── explain_graphs/
│   ├── probability_extraction.png
│   ├── vn_entropy.png                  # entropía de von Neumann por sitio
│   ├── mutual_information.png          # matriz de información mutua entre sitios
│   ├── feature_importance.png
│   ├── family_feature_importance.png
│   ├── anomaly_breakdown.png
│   ├── bond_entropy.png                # entropía de entrelazamiento por enlace
│   ├── conditional_probabilities.png
│   └── joint_probabilities.png
│
└── explain_tables/
    ├── probability_extraction.csv
    ├── vn_entropy.csv
    ├── mutual_information.csv
    ├── feature_importance.csv
    ├── family_feature_importance.csv
    ├── anomaly_breakdown.csv
    ├── bond_entropy.csv
    ├── conditional_probabilities.csv
    └── joint_probabilities.csv
```

Cada figura está envuelta de forma que un CSV ausente o mal formado solo omite esa
figura concreta, sin abortar toda la ejecución.

---

## Configuración del entrenamiento

Los hiperparámetros viven en `train_mps_nsl_kdd.py` (constantes de cabecera +
`DMRGConfig`). Algunos relevantes:

| Parámetro | Valor por defecto | Significado |
|---|---|---|
| `DTYPE` | `float64` | Precisión de los tensores del MPS |
| `INIT_BOND_DIM` | `2` | Bond dimension inicial |
| `VAL_FRACTION` | `0.15` | Fracción normal reservada para validación/umbral |
| `num_loops` | `150` | Número de barridos DMRG |
| `max_bond_dim` | `128` | Tope de la bond dimension |
| `bond_growth_factor` | `1.5` | Factor de crecimiento adaptativo del enlace |
| `discarded_weight_threshold` | `1e-3` | Umbral de peso descartado para crecer |
| `lr` / `lr_min` | `8e-4` / `5e-5` | Tasa de aprendizaje y su mínimo |
| `early_stopping_patience` | `15` | Paciencia de parada temprana |
| `batch_size` | `1024` | Tamaño de minibatch |
| `metric_for_stopping` | `val_nll` | Métrica monitorizada |
| `seed` | `123` | Reproducibilidad |

El parámetro `target_d_numeric` del codificador (en `encoder_nsl_kdd.py`,
`main()`) controla cuántos bins cuantil se usan para las características numéricas.

---

## Notas

- El MPS usa contracción con reescalado por sitio y *clamping* dependiente de
  `dtype`, por lo que `float64` es preferible para estabilidad numérica con
  muchos sitios.
- El esquema de codificación (`encoding_schema.json`) es la fuente de verdad de
  las dimensiones físicas; el entrenador valida que cada columna caiga en
  `[0, d_k)` antes de empezar.
- Para muestrear desde el modelo entrenado, carga el MPS con `MPS.load(...)` y usa
  `MPSSampler` de `mps_generative.py`.
