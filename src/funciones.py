import pandas as pd
import numpy as np
import joblib

variables=[
    'A', 'B', 'C', 'D', 'E', 'H', 'J', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S','Monto'
    ]

def seleccion_variables(
        df:pd.DataFrame,
        lista_vars: list = variables) -> pd.DataFrame:
    """
    Selecciona las columnas indicadas de un DataFrame. En particular las utilizadas 
    para el funcionamiento del modelo de fraude.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        lista_vars (list): Lista de nombres de columnas a conservar.

    Returns:
        pd.DataFrame: DataFrame con solo las columnas seleccionadas.

    Raises:
        KeyError: Si alguna columna no se encuentra en el DataFrame.
    """

    try:
        df = df[lista_vars]
        return df
    except KeyError as e:
        raise KeyError(f"Al menos un campo no pertenece al conjunto de variables: {e}")

    

def limpiar_y_convertir(
        df:pd.DataFrame, 
        columnas:list) -> pd.DataFrame :
    """
    Limpia separadores de miles y convierte columnas numéricas a float.
    Ejemplo de valores corregidos: '1,234.8' → 1234.8

    Parámetros:
    -----------
    df : pd.DataFrame, DataFrame con las columnas a limpiar
    columnas : list, Lista de nombres de columnas a transformar

    Retorna:
    --------
    pd.DataFrame con las columnas convertidas a float
    """
    for col in columnas:
        df[col] = (
            df[col]
            .astype(str)                     # convierte todo a string
            .str.replace(',', '', regex=False)  # elimina comas (miles)
            .str.strip()                     # elimina espacios extra
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convierte a float (NaN si no puede)
    return df

def reemplazar_negativos_por_mediana(
        df: pd.DataFrame, 
        columnas: list) -> pd.DataFrame:
    """
    Reemplaza los valores negativos en las columnas indicadas por la mediana
    de su respectiva distribución (calculada sobre toda la columna).

    Parámetros
    ----------
    df : pd.DataFrame, DataFrame con las columnas a corregir.
    columnas : list, Lista con los nombres de las columnas a procesar.

    Retorna
    -------
    pd.DataFrame con los valores negativos reemplazados por la mediana.
    """
    for col in columnas:
        mediana = df[col].median(skipna=True)
        df[col] = np.where(df[col] < 0, mediana, df[col]) #Reemplazo
    return df

def imputar_nulos_por_mediana(
        df: pd.DataFrame, 
        columnas: list
        ) -> pd.DataFrame:
    """
    Imputa los valores nulos en las columnas indicadas usando la mediana
    de cada variable.

    Parámetros
    ----------
    df : pd.DataFrame, DataFrame con las columnas a corregir.
    columnas : list, Lista con los nombres de las columnas a procesar.

    Retorna
    -------
    pd.DataFrame con los valores nulos imputados por la mediana.
    """
    for col in columnas:
        mediana = df[col].median(skipna=True)
        df[col] = df[col].fillna(mediana)
    return df

def dummificar_var_categorica(
        df:pd.DataFrame, 
        columna:str, 
        categorias_principales:list, 
        referencia:str='OTROS') -> pd.DataFrame:
    """
    Agrupa las categorías no deseadas en 'OTROS', crea variables dummies
    y elimina la categoría de referencia.

    Parámetros
    ----------
    df : pd.DataFrame, DataFrame de entrada.

    columna : str,  Nombre de la columna categórica a transformar.

    categorias_principales : list, Lista de categorías que se desean conservar tal cual.

    referencia : str, opcional (por defecto = 'OTROS') Categoría que se tomará como referencia (no se crea dummy para ella).

    Retorna
    -------
    pd.DataFrame
        DataFrame con las categorías agrupadas y variables dummies creadas.
    """

    df = df.copy()

    # Agrupación de categorías en otros 
    df[columna] = np.where(df[columna].isin(categorias_principales), df[columna], 'OTROS')

    # Se crean las columnas dummies de las categorías
    categorias = [cat for cat in df[columna].unique() if cat != referencia]
    for cat in categorias:
        df[f"{columna}_{cat}"] = np.where(df[columna] == cat, 1, 0)

    # Se elimina el campo original
    df = df.drop(columns=columna)

    return df

def binarizar_variables(
        df: pd.DataFrame, 
        vars_corte_0:list, 
        vars_corte_1:list) -> pd.DataFrame:
    """
    Binariza variables según su umbral:
    - Para las variables en 'vars_corte_0', asigna 1 si valor > 0, sino 0.
    - Para las variables en 'vars_corte_1', asigna 1 si valor > 1, sino 0.

    Parámetros
    ----------
    df : pd.DataFrame. DataFrame con las variables numéricas.
    
    vars_corte_0 : list, Lista de variables cuyo umbral de comparación es 0.
    
    vars_corte_1 : list, Lista de variables cuyo umbral de comparación es 1.

    Retorna
    -------
    pd.DataFrame con las variables transformadas a 0/1.
    """

    df = df.copy()

    # Variables con valor mínimo en 0
    for col in vars_corte_0:
        if col in df.columns:
            df[col] = np.where(df[col] > 0, 1, 0)

    # Variables con valor mínimo en 1
    for col in vars_corte_1:
        if col in df.columns:
            df[col] = np.where(df[col] > 1, 1, 0)

    return df

def transformacion_raiz_cuadrada(
        df: pd.DataFrame, 
        variables:list) -> pd.DataFrame:
    """
    Aplica la transformación raíz cuadrada a las columnas especificadas.
    Los valores negativos se reemplazan por NaN antes de aplicar la raíz.

    Parámetros
    ----------
    df : pd.DataFrame, DataFrame de entrada.
    variables : list, Lista de nombres de columnas a transformar.

    Retorna
    -------
    pd.DataFrame, con las columnas transformadas.
    """
    
    df=df.copy()

    for col in variables:
        df[col] = np.where(df[col] >= 0, np.sqrt(df[col]), np.nan)

    return df

def cargar_modelo(ruta_modelo: str):
    """
    Carga un modelo previamente guardado en formato .joblib.

    Parámetros
    ----------
    ruta_modelo : str, Ruta completa del archivo .joblib

    Retorna
    -------
    modelo : object Modelo cargado listo para usar.
    """
    try:
        modelo = joblib.load(ruta_modelo)
        print(f"Modelo cargado correctamente desde: {ruta_modelo}")
        return modelo
    except FileNotFoundError:
        print(f"No se encontró el archivo en la ruta: {ruta_modelo}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

def ejecutar_modelo(
        modelo, 
        df: pd.DataFrame,
        umbral: float = 0.565) -> pd.DataFrame:
    """
    Aplica un modelo de clasificación binaria sobre una base de datos
    y agrega las columnas con la probabilidad y la predicción ajustada
    según un umbral de clasificación.

    Parámetros
    ----------
    modelo : object, Modelo previamente entrenado (debe tener el método predict_proba).
    data : pd.DataFrame, Conjunto de datos sobre el cual se generarán las predicciones.
    umbral : float, opcional (default=0.565), Umbral de clasificación para determinar la clase positiva (Fraude=1).

    Retorna
    -------
    pd.DataFrame
        DataFrame original con dos nuevas columnas:
        - 'PROB_FRAUDE': Probabilidad de fraude
        - 'PRED_FRAUDE': Predicción binaria (0 o 1)
    """
    try:
        # Predicción de probabilidades
        df = df.copy()
        df["PROB_FRAUDE"] = modelo.predict_proba(df)[:, 1]

        # Predicción según el umbral
        df["PRED_FRAUDE"] = np.where(df["PROB_FRAUDE"] >= umbral, 1, 0)

        print(f"Predicciones generadas correctamente (umbral = {umbral})")
        return df

    except Exception as e:
        print(f"Error al ejecutar el modelo: {e}")
        return df