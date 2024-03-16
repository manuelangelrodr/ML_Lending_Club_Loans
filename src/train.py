# SCRIPT DE ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING

# IMPORTAMOS LIBRERIAS

# Importamos las librerías Python que vamos a necesitar en nuestro estudio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE


import pickle

import sys
sys.path.append('../')
from utils.eda_functions import *

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


import pickle

import sys
sys.path.append('../')
from utils.eda_functions import *

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')



# FUNCIONES AUXILIARES

def cargar_dataset(ruta_archivo):
    """
    Carga un conjunto de datos utilizando Pandas desde un archivo CSV.

    Argumentos:
    ruta_archivo (str): Ruta del archivo CSV a cargar.

    Retorna:
    pd.DataFrame: El DataFrame que contiene los datos cargados desde el archivo CSV.
    """
    try:
        dataframe = pd.read_csv(ruta_archivo)
        return dataframe
    except FileNotFoundError:
        print("El archivo no se encontró en la ruta especificada.")
        return None
    except Exception as e:
        print("Ocurrió un error al cargar el dataset:", e)
        return None


# UNIMOS LOS DATASETS
    
def unir_datasets_por_id(dataset1, dataset2, columna_id):
    """
    Une dos conjuntos de datos utilizando Pandas por una columna de identificación común.

    Argumentos:
    dataset1 (pd.DataFrame): El primer DataFrame a unir.
    dataset2 (pd.DataFrame): El segundo DataFrame a unir.
    columna_id (str): El nombre de la columna de identificación.

    Retorna:
    pd.DataFrame: El DataFrame resultante de la unión de los dos conjuntos de datos por la columna de identificación.
    """
    try:
        resultado = pd.merge(dataset1, dataset2, on=columna_id)
        return resultado
    except Exception as e:
        print("Ocurrió un error al unir los datasets:", e)
        return None


# ELIMINAMOS COLUMNAS DE UN LISTA QUE NO APORTAN AL MODELO
    
def eliminar_columnas(dataset, columnas_a_eliminar):
    """
    Elimina las columnas especificadas de un conjunto de datos.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame del que se eliminarán las columnas.
    columnas_a_eliminar (list): Una lista de nombres de columnas a eliminar.

    Retorna:
    pd.DataFrame: El DataFrame resultante después de eliminar las columnas especificadas.
    """
    try:
        resultado = dataset.drop(columns=columnas_a_eliminar, inplace=True)
        return resultado
    except Exception as e:
        print("Ocurrió un error al eliminar las columnas:", e)
        return None

# CAMBIAR NOMBRE A LAS COLUMNAS DE UNA LISTA
    
def cambiar_nombres_columnas(dataset, diccionario_nombres):
    """
    Cambia los nombres de las columnas de un conjunto de datos según los valores proporcionados en un diccionario.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame al que se cambiarán los nombres de las columnas.
    diccionario_nombres (dict): Un diccionario donde las claves son los nombres de las columnas actuales
    y los valores son los nuevos nombres de las columnas.

    Retorna:
    pd.DataFrame: El DataFrame con los nombres de las columnas modificados según el diccionario.
    """
    try:
        dataset_renombrado = dataset.rename(columns=diccionario_nombres, inplace=True)
        return dataset_renombrado
    except Exception as e:
        print("Ocurrió un error al cambiar los nombres de las columnas:", e)
        return None




# TRATAMOS LOS VALORES MISSING
    
def rellenar_nans_con_cero(dataset, columna):
    """
    Rellena los valores NaN en una columna específica de un conjunto de datos con el valor 0.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame en el que se rellenarán los valores NaN.
    columna (str): El nombre de la columna en la que se rellenarán los valores NaN.

    Retorna:
    pd.DataFrame: El DataFrame con los valores NaN en la columna especificada rellenados con 0.
    """
    try:
        dataset[columna].fillna(0, inplace=True)
        return dataset
    except Exception as e:
        print("Ocurrió un error al rellenar los valores NaN con cero:", e)
        return None
    

def eliminar_nans_en_columna(dataset, columna):
    """
    Elimina las filas que contienen valores NaN en una columna específica de un conjunto de datos.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame del que se eliminarán las filas con valores NaN.
    columna (str): El nombre de la columna en la que se buscarán valores NaN.

    Retorna:
    pd.DataFrame: El DataFrame resultante después de eliminar las filas con valores NaN en la columna especificada.
    """
    try:
        dataset_sin_nans = dataset.dropna(subset=[columna], inplace=True)
        return dataset_sin_nans
    except Exception as e:
        print("Ocurrió un error al eliminar los valores NaN de la columna:", e)
        return None



# PASAMOS LA VARIABLE 'annual_income' A ESCALA LOGARITMICA
    
def escala_logaritmica(dataset, columna):
    """
    Transforma una columna específica de un conjunto de datos a escala logarítmica.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame en el que se transformará la columna.
    columna (str): El nombre de la columna que se transformará.

    Retorna:
    pd.DataFrame: El DataFrame con la columna especificada transformada a escala logarítmica.
    """
    try:
        dataset[columna] = np.log(dataset[columna])
        return dataset
    except Exception as e:
        print("Ocurrió un error al transformar la columna a escala logarítmica:", e)
        return None


# TRATAMOS LOS OUTLIERS

def igualar_valores_extremos_iqr(dataset, columna, veces_iqr):
    """
    Iguala los valores menores y mayores a un cierto número de veces el rango intercuartílico (IQR)
    de una variable específica en un conjunto de datos.

    Argumentos:
    dataset (pd.DataFrame): El DataFrame en el que se igualarán los valores extremos.
    columna (str): El nombre de la columna en la que se calculará el rango intercuartílico.
    veces_iqr (int): El número de veces el rango intercuartílico para igualar los valores extremos.

    Retorna:
    pd.DataFrame: El DataFrame con los valores extremos igualados.
    """
    try:
        q1 = dataset[columna].quantile(0.25)
        q3 = dataset[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - veces_iqr * iqr
        limite_superior = q3 + veces_iqr * iqr

        dataset[columna] = dataset[columna].clip(lower=limite_inferior, upper=limite_superior)
        
        return dataset
    except Exception as e:
        print("Ocurrió un error al igualar los valores extremos:", e)
        return None


# DIVIDIMOS LOS DATASETS EN TRAIN Y TEST
    
def dividir_dataset(df, test_size=0.25, random_state=None, target='loan_status', no_incluir='id'):
    """
    Función para dividir un DataFrame en conjuntos de entrenamiento y prueba.

    Args:
    - df: DataFrame que se desea dividir.
    - test_size: Porcentaje del dataset que se desea asignar al conjunto de prueba. Por defecto, es 0.25.
    - random_state: Semilla para la generación de números aleatorios. Por defecto, es None.

    Returns:
    - train_set: DataFrame de entrenamiento.
    - test_set: DataFrame de prueba.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    X_train_set = train_set.drop(columns=[no_incluir, target])
    X_test_set = test_set.drop(columns=[no_incluir, target])
    y_train_set = train_set[target]
    y_test_set = test_set[target]
    return X_train_set, X_test_set, y_train_set, y_test_set


# HACEMOS LA TRANSFORMACION DE VARIABLES

def preprocess_data(X_train, X_test, numeric_cols, categorical_cols):
    """
    Preprocesa los datos de entrenamiento y prueba utilizando un pipeline que incluye 
    escalado estándar para variables numéricas y codificación OneHot para variables categóricas.

    Argumentos:
    X_train (pd.DataFrame): Conjunto de datos de entrenamiento.
    X_test (pd.DataFrame): Conjunto de datos de prueba.
    numeric_cols (list): Lista de nombres de columnas numéricas.
    categorical_cols (list): Lista de nombres de columnas categóricas.

    Retorna:
    pd.DataFrame, pd.DataFrame: Conjuntos de datos de entrenamiento y prueba preprocesados.
    """
    # Crear el preprocesador para las columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Crear el pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Aplicar el pipeline al conjunto de datos de entrenamiento y prueba
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Obtener los nombres de las columnas después de la transformación OneHotEncoder
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)

    # Combinar los nombres de columnas numéricas y categóricas
    transformed_feature_names = list(numeric_cols) + list(ohe_feature_names)

    # Balanceamos el dataset de train con los el método SMOTE

    smote = SMOTE()
    X_train_transformed_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

    X_train_transformed = X_train_transformed_resampled.copy()
    y_train = y_train_resampled.copy()


    # Convertir la salida a DataFrame de pandas
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)

    return X_train_transformed_df, X_test_transformed_df

# Misma funcion que el anterior pero sin datos de prueba

def dividir_dataset(df, test_size=0.25, random_state=None, target='loan_status', no_incluir='id'):
    """
    Función para dividir un DataFrame en conjuntos de entrenamiento y prueba.

    Args:
    - df: DataFrame que se desea dividir.
    - test_size: Porcentaje del dataset que se desea asignar al conjunto de prueba. Por defecto, es 0.25.
    - random_state: Semilla para la generación de números aleatorios. Por defecto, es None.

    Returns:
    - train_set: DataFrame de entrenamiento.
    - test_set: DataFrame de prueba.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    X_train_set = train_set.drop(columns=[no_incluir, target])
    X_test_set = test_set.drop(columns=[no_incluir, target])
    y_train_set = train_set[target]
    y_test_set = test_set[target]
    return X_train_set, X_test_set, y_train_set, y_test_set


# HACEMOS LA TRANSFORMACION DE VARIABLES

def pipeline_preprocess_data(df, numeric_cols, categorical_cols):
    """
    Preprocesa los datos utilizando un pipeline que incluye 
    escalado estándar para variables numéricas y codificación OneHot para variables categóricas.

    Argumentos:
    df (pd.DataFrame): Conjunto de datos .
    
    numeric_cols (list): Lista de nombres de columnas numéricas.
    categorical_cols (list): Lista de nombres de columnas categóricas.

    Retorna:
    pd.DataFrame, : Conjuntos de datos dpreprocesados.
    """
    # Crear el preprocesador para las columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Crear el pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Aplicar el pipeline al conjunto de datos de entrenamiento y prueba
    df_transformed = pipeline.fit_transform(df)
    

    # Obtener los nombres de las columnas después de la transformación OneHotEncoder
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)

    # Combinar los nombres de columnas numéricas y categóricas
    transformed_feature_names = list(numeric_cols) + list(ohe_feature_names)

    # Convertir la salida a DataFrame de pandas
    df_transformed_df = pd.DataFrame(df_transformed, columns=transformed_feature_names)
   

    return df_transformed_df



# Devuelve las metricas del dataset de entrenamiento

def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
    model_name (string): El modelo del que queremos las métricas
    model_object: a fit GridSearchCV object
    metric (string): roc_auc, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, auc_roc, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'roc_auc': 'mean_test_roc_auc',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 'precision' : 'mean_test_precision'
                 }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    roc_auc = best_estimator_results.mean_test_roc_auc
    accuracy = best_estimator_results.mean_test_accuracy
    precision = best_estimator_results.mean_test_precision

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                        'roc_auc': [roc_auc],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        'precision' : [precision]
                        })

    return table

# Devuelve las metricas del dataset de prueba
def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
    model_name (string): Your choice: how the model will be named in the output table
    preds: numpy array of test predictions
    y_test_data: numpy array of y_test data

    Out:
    table: a pandas df of roc_auc, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    roc_auc = roc_auc_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                        'roc_auc': [roc_auc],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        'precision' : [precision]
                        })

    return table

# Guarda el modelo entrenado
def write_pickle(path, model_object, save_name:str):
    '''
    save_name is a string.
    '''
    with open(path + save_name + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)

# Recuperar el modelo guardado
        
def read_pickle(path, saved_model_name:str):
    '''
    saved_model_name is a string.
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

        return model


# TRATAMIENTO PREVIO DEL DATASET

df_1 = cargar_dataset('../data/raw/loans-part-1.csv')
df_2 = cargar_dataset('../data/raw/loans-part-2.csv')

df_original = unir_datasets_por_id(df_1, df_2, 'id')

df = df_original.copy()

columnas_eliminar = ['funded_amount_by_investors', 'issued_on', 'employer_title', 'sub_grade']
eliminar_columnas(df, columnas_eliminar)

columns={'revolving_line_utilization_rate': 'revolving_rate', 'inquiries_last_6_months': 'inquiries', 'derogatory_public_records': 'derogatory'}
cambiar_nombres_columnas(df, columns)

rellenar_nans_con_cero(df, 'revolving_rate')

eliminar_nans_en_columna(df, 'employment_length')

escala_logaritmica(df, 'annual_income')

igualar_valores_extremos_iqr(df, 'annual_income', 4)

eliminar_columnas(df, ['installment', 'open_credit_lines'])

df = df[df['loan_status'] != 'current']

df['loan_status'] = df['loan_status'].map({'charged off': 1, 'fully paid': 0})

X_train, X_test, y_train, y_test = dividir_dataset(df, test_size=0.25, random_state=42)

# Definir las columnas categóricas y numéricas
categorical_cols = ['grade', 'verification_status', 'purpose', 'home_ownership']
numeric_cols = ['loan_amount', 'loan_term', 'interest_rate', 'dti', 'inquiries',
                'derogatory', 'revolving_rate', 'total_credit_lines', 'employment_length', 'annual_income']



X_train_transformed_df, X_test_transformed_df = preprocess_data(X_train, X_test, numeric_cols, categorical_cols)



# MODELO REGRESION LOGISTICA

lr = LogisticRegression()
lr.fit(X_train_transformed_df, y_train)

y_pred = lr.predict(X_test_transformed_df)

# 1. Instantiate LogisticRegression
lr = LogisticRegression(random_state=42, class_weight='balanced')

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'penalty': [None, 'l1', 'l2'],
             'C': [0.1, 0.5, 1],
             'solver' : ['saga', 'lbfgs'],
            }

# 3. Define a dictionary of scoring metrics to capture
scoring = ['accuracy', 'roc_auc', 'recall', 'f1', 'precision']

# 4. Instantiate the GridSearchCV object
lr_cv = GridSearchCV(lr, cv_params, scoring=scoring, cv=5, refit='recall')

lr_cv.fit(X_train_transformed_df, y_train)

lr_cv_results = make_results('Logistic Regression CV', lr_cv, 'recall')

lr_val_preds = lr_cv.best_estimator_.predict(X_test_transformed_df)

lr_val_scores = get_test_scores('LogisticRegression val', lr_val_preds, y_test)

results = pd.concat([lr_cv_results, lr_val_scores], axis=0)
print(results)

# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, lr_val_preds, labels=lr_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['fully paid', 'charged off'])
disp.plot();

path = '../model/'

write_pickle(path, lr_cv, 'lr')







