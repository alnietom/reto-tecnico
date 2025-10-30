# Detección de Fraude en Transacciones — Clasificación Binaria
📘 Contexto

El fraude en plataformas de pagos es una problemática mundial que representa pérdidas económicas significativas.
Por su alcance y volumen de transacciones, es común el enfrentamiento diario a intentos de fraude en diferentes etapas del flujo de pagos.

El objetivo de este proyecto es construir un modelo de machine learning robusto y escalable para la detección temprana de fraudes, capaz de distinguir entre transacciones legítimas y fraudulentas con alta precisión.

# Estructura del Proyecto

```bash
├── data/
│   ├── MercadoLibre.csv        # Datos crudos
│   └── MatrizDatos.csv         # Datos procesados listos para modelar
│
├── notebooks/
│   ├── 01_analisis_descriptivo.ipynb   # Exploración inicial de los datos
│   ├── 02_preprocesamiento.ipynb       # Limpieza, transformación y generación de variables
│   └── 03_ajuste_modelo_fraude.ipynb   # Entrenamiento y evaluación del modelo
│
├── src/
│   └── funciones.py            # Funciones auxiliares para el preprocesamiento y predicción
│
├── model/
│   └── modelo_fraude.joblib    # Modelo final entrenado
│
├── main.ipynb                  # Notebook final integrador del flujo completo
├── requirements.txt            # Dependencias del entorno
└── README.md                   # Documentación del proyecto
```
# Ejecución del Proyecto
## Opción 1 — Notebook principal

Ejecutar el flujo completo de análisis, modelado y predicción desde:

main.ipynb

## Opción 2 — Flujo modular por etapas

-01_analisis_descriptivo.ipynb: análisis exploratorio de los datos.

-02_preprocesamiento.ipynb: limpieza, codificación, escalado y creación de la matriz final.

-03_ajuste_modelo_fraude.ipynb: entrenamiento, ajuste de hiperparámetros y evaluación del modelo.

# Modelo de Machine Learning

El modelo entrenado (modelo_fraude.joblib) es un clasificador binario que predice la probabilidad de que una transacción sea fraudulenta (1) o no fraudulenta (0).

Se evalúan métricas de desempeño tales como:

- AUC-ROC

- Precisión

- Recall

- F1-score

Además, se determina el umbral óptimo de clasificación a partir de la maximización de la ganancia obteniendo un valor de 0,36.
