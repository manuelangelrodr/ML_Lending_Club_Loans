import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu


# FUNCIONES ANÁLISIS UNIVARIANTE

def graficos_var_cont(df, column_df, color):
    '''
    Función que crea dos gráficos:
    A la izquierda se crea un histograma de una variable continua con la función de densidad
    A la derecha se crea un histograma de la variable continua discriminando si el préstamo se ha pagado o no

    Parámetros:
    df: Dataframe de Pandas
    column_df: Nombre de la columna de la variable continua
    color: Color de los histogramas

    Output:
    Dos gráficos de tipo histograma

    '''
      

    # Creamos la figura y los subplots en 1 fila y 2 columnas   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Primer gráfico: histograma con función de densidad
    sns.histplot(df[column_df], kde=True, bins=20, color = color, ax=ax1)
    ax1.set_xlabel(f'{column_df}')
    ax1.set_ylabel('Densidad')
    ax1.set_title(f'Histograma de la columna {column_df} y función de densidad')

    # Segundo gráfico: histograma de la variable de estudio por 'loan_status'
    status_counts = df['loan_status'].value_counts(normalize=True) * 100

    for status in df['loan_status'].unique():
        loan_amounts = df[df['loan_status'] == status][column_df]
        ax2.hist(loan_amounts, bins=20, alpha=0.5, label=f"{status} ({status_counts[status]:.2f}%)")

    ax2.set_xlabel(f'{column_df}')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(f'Histograma de la columna {column_df} según si el préstamo se ha pagado o no')
    ax2.legend()

    # Ajustar espaciado entre subplots
    plt.tight_layout()

    # Mostrar los gráficos combinados
    plt.show()

def mapa_calor_impagados_cont(df, column_df, color):

    '''
    Función que crea un mapa de calor de una columna con variables contínuas 
    según el porcentaje de impagados por cada tramo.
    La función divide los tramos en 20 tramos.

    Parámetros:
    df: Dataframe de Pandas
    col_heatmap: Nombre de la columna de la variable continua
    color: Color del mapa de calor

    Output:
    Mapa de calor
    '''
    # Copiamos el df
    df_copy = df.copy()
    
    # Divide la variable en 20 tramos
    df_copy['bins'] = pd.cut(df_copy[column_df], bins=20)

    
    # Filtrar datos donde 'loan_status' sea igual a 'charged off'
    charged_off_data = df_copy[df['loan_status'] == 'charged off']
    fully_paid_data = df_copy[df['loan_status'] == 'fully paid']

    # Calcular los porcentajes de 'charged off' para cada tramo de la variable
    pivot_data_impagados = charged_off_data['bins'].value_counts().sort_index()
    pivot_data_pagados = fully_paid_data['bins'].value_counts().sort_index()
    pivot_data = round((pivot_data_impagados / pivot_data_pagados) * 100, 2) 

    # Crear un mapa de calor
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data.to_frame(), cmap=color, annot=True, fmt='.2f', cbar=True)
    plt.title(f'Porcentaje de impagados por tramo de {column_df}')
    plt.xlabel(f'Porcentajes de impagados de {column_df}')
    plt.ylabel(f'Tramos de {column_df}')
    plt.tight_layout()
    plt.show()



def grafico_mapa_calor_impagados(df, col_heatmap, color):

    '''
    Función que crea un mapa de calor de una columna con variables paramétricas 
    según el porcentaje de impagados por cada categoría

    Parámetros:
    df: Dataframe de Pandas
    col_heatmap: Nombre de la columna de la variable paramétrica
    color: Color del mapa de calor

    Output:
    Mapa de calor
    '''
    
    # Filtrar el DataFrame para 'loan_status' igual a 'charged off'
    charged_off_df = df[df['loan_status'] == 'charged off']

    # Contamos los valores de cada categoría de la columna de estudio con impagados
    grouped = charged_off_df.groupby(col_heatmap)['loan_status'].count()

    # Calcular los porcentajes relativos para cada valor de la columna de estudio
    relative_percentages = (grouped / df[col_heatmap].value_counts()) * 100

    # Crear un DataFrame con los porcentajes relativos
    relative_percentages_df = pd.DataFrame(relative_percentages, columns=['Charged Off Relative Percentage'])

    # Crear un gráfico de calor
    plt.figure(figsize=(10, 6))
    heatmap = plt.imshow(relative_percentages_df.values.reshape(1, -1), aspect='auto', cmap= color)

    # Añadir etiquetas, título y colorbar
    plt.xticks(range(len(relative_percentages_df)), relative_percentages_df.index, rotation=90)
    plt.xlabel(col_heatmap)
    plt.ylabel('')
    plt.title(f'Porcentaje relativo de impagados de {col_heatmap}')
    plt.colorbar(heatmap)

    plt.tight_layout()
    plt.show()



def grafico_barras_loan_status(df, column_df):
    '''
    Función que crea un gráfico de barras con las variables categóticas distinguiendo si el crédito se ha pagado o no

    Parámetros:
    df: Dataframe de Pandas
    column_df: Nombre de la columna de la variable de estudio
    

    Output:
    Gráfico de barras de la variable categórica distinguiendo entre pagado e impagado
    '''

    # Obtener el recuento de combinaciones 'purpose'-'loan_status'
    grouped = df.groupby([column_df, 'loan_status']).size().unstack()

    # Crear el gráfico de barras apiladas
    fig, ax = plt.subplots(figsize=(10, 6))

    index = range(len(grouped))

    # Iterar sobre cada tipo de 'purpose' y dibujar barras apiladas para cada tipo de 'loan_status'
    bottoms = [0] * len(grouped)
    for status in grouped.columns:
        ax.bar(index, grouped[status], label=status, bottom=bottoms)
        bottoms = [sum(x) for x in zip(bottoms, grouped[status])]

    # Añadir etiquetas, título y leyenda
    ax.set_xlabel(column_df)
    ax.set_ylabel('Valores')
    ax.set_title(f'{column_df} por Pagados/Impagados')
    ax.set_xticks(index)
    ax.set_xticklabels(grouped.index, rotation=90)
    ax.legend()

    # Mostrar el gráfico de barras apiladas
    plt.tight_layout()
    plt.show()


# FUNCION CALCULO PORCENTAJES VARIABLE CATEGORICA POR STATUS DEL PRÉSTAMO

def porcentaje_var_cat_status(df, column_df):
    '''
    Función que calcula el porcentaje de cada valor de la variable categórica por cada valor de 'loan_status'

    Parámetros:
    df: Dataframe de Pandas
    column_df: Nombre de la columna de la variable categóricade estudio
    

    Output:
    Porcentaje de cada valor de la variable categórica por cada valor de 'loan_status'
    '''
    # Contar los valores de la variable categórica por 'loan_status'
    grouped = df.groupby('loan_status')[column_df].value_counts(normalize=True).unstack() * 100

    return grouped


# FUNCIONES TESTS ESTÁDISTICOS

# Función Prueba T-Student


def test_t_student(df_paid, df_charged_off, column_df):
    '''
    La función realiza la prueba t-Student entre una variable numérica continua de la que no conocemos su distribución
    o sabemos que no es una gaussiana y la otra variable categórica 'loan_status' que indica si el prestatmo se ha pagado o no.

    Parámetros:
    df_paid: Dataframe de Pandas con los prestatmos pagados
    df_charged_off: Dataframe de Pandas con los prestatmos impagados
    column_df: Nombre de la columna de la variable categórica de estudio
    
    Output:
    Estadístico t, valor p, rechazo o no de la hipótesis nula
    '''


    # Realizar la prueba de t de Student para comparar las medias de 'loan_amount' entre 'fully paid' y 'charged off'
    t_statistic, p_value = ttest_ind(df_paid[column_df], df_charged_off[column_df], equal_var=False)

    # Imprimir el resultado del contraste de hipótesis
    print(f"Estadístico t: {t_statistic}")
    print(f"Valor p: {p_value}")
    if p_value < 0.05:
        print(f"Se rechaza la hipótesis nula: La variable {column_df} incide en el impago del crédito")
    else:
        print(f"No se puede rechazar la hipótesis nula: la variable {column_df} no incide en el impago del crédito")


# Función Prueba Chi-Cuadrado
def test_chi_cuadrado(df, column_df):
    '''
    La función realiza la prueba de chi-cuadrado entre una variable categórica de estudio
    y la otra variable categórica 'loan_status' que indica si el prestatmo se ha pagado o no.

    Parámetros:
    df: Dataframe de Pandas
    column_df: Nombre de la columna de la variable categórica de estudio
    
    Output:
    Estadístico Chi-cuadrado, valor p, rechazo o no de la hipótesis nula
    '''
    # Crear una tabla de contingencia entre 'home_ownership' y 'loan_status'
    contingency_table = pd.crosstab(df[column_df], df['loan_status'])

    # Realizar la prueba de chi-cuadrado
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Imprimir los resultados del contraste de hipótesis
    print(f"Estadístico Chi-cuadrado: {chi2}")
    print(f"Valor p: {p_value}")
    if p_value < 0.05:
        print(f"Se rechaza la hipótesis nula: la variable {column_df} incide en en el impago del crédito")
    else:
        print(f"No se puede rechazar la hipótesis nula: la variable {column_df} no incide en el impago del crédito")

def test_mann_whitney(df1, df2, column_df):
    '''
    La función realiza la prueba de Mann-Whitney entre dos variable numérica continua de la que no conocemos su distribución.


    Parámetros:
    df: Dataframe de Pandas
    column_df: Nombre de la columna de la variable numérica
    
    Output:
    Estadístico Kruskal-Wallis, valor p, rechazo o no de la hipótesis nula
    '''

    # Realizar la prueba de Mann-Whitney
    resultado_mannwhitney = mannwhitneyu(df1[column_df],
                                         df2[column_df])

    # Imprimir el resultado del contraste de hipótesis
    print(f"Estadístico Mann-Whitney: {resultado_mannwhitney.statistic}")
    print(f"Valor p: {resultado_mannwhitney.pvalue}")
    if resultado_mannwhitney.pvalue < 0.05:
        print(f"Se rechaza la hipótesis nula: Hay diferencias significativas de la variable {column_df} entre pagados e impagados")
    else:
        print(f"No se puede rechazar la hipótesis nula: No hay diferencias significativas de la variable {column_df} entre pagos e impagos")


# FUNCION DATA-REPORT
        
def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

