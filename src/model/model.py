# IMPORTAR LIBRERIAS
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import pickle
import os

# CARGAMOS EL DATASET

file_path = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\data\processed\df_preparadol.csv'

# import sys
# sys.path.append('../')
# from utils.eda_functions import *

# Cargar el archivo CSV en un DataFrame de pandas
df = pd.read_csv(file_path)

# CONVERTIMOS LA VARIABLE 'annual_income' a escala logaritmica

df['annual_income'] = np.log(df['annual_income'] + 1)

pickle_filename = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\checkpoints\01_log.pkl'

with open(pickle_filename, 'wb') as file:
    pickle.dump(df, file)

# ELIMINAMOS LAS COLUMNAS 'installment' y 'open_credit_lines'

df.drop(columns=['installment', 'open_credit_lines'], inplace=True)


# MAPEO DE LA VARIABLE OBJETIVO
    
df['loan_status'] = df['loan_status'].map({'fully paid': 0, 'charged off': 1})

clean_pickle_filename = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\checkpoints\02_dataset_mapeado.pkl'

with open(clean_pickle_filename, 'wb') as file:
    pickle.dump(df, file)

# PREPARACION DE LOS DATOS
    
# Crear datasets X e y
X = df.drop(columns=['loan_status', 'id'])
y = df['loan_status']

X.to_csv(r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\data\processed\X.csv', index=False)
y.to_csv(r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\data\processed\y.csv', index=False)
    
# Definir las columnas categóricas y numéricas
categorical_cols = ['grade', 'verification_status', 'purpose', 'home_ownership']
numeric_cols = ['loan_amount', 'loan_term', 'interest_rate', 'dti', 'inquiries',
                'derogatory', 'revolving_rate', 'total_credit_lines', 'employment_length', 'annual_income']

# TRANSFORMAR LAS VARIABLES CATEGORICAS Y NUMERICAS

# Crear el transformador de columnas con OneHotEncoder y StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Ajustar el preprocesador a los datos
preprocessor.fit(X)

# Guardar el preprocesador en un archivo pickle
preprocessor_filename = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\checkpoints\03_preprocessor.pkl'
with open(preprocessor_filename, 'wb') as file:
    pickle.dump(preprocessor, file)

# print(df.head())

# ENTRENAR EL MODELO

# Crear el modelo de regresión logística con los hiperparámetros especificados
log_reg_model = LogisticRegression(class_weight='balanced', penalty='l2', C=0.1, solver='saga', max_iter=1000)

# Crear un pipeline que incluya el preprocesamiento y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', log_reg_model)])

# Entrenar el modelo con los datos de entrenamiento
pipeline.fit(X, y)

# Guardar el modelo entrenado en un archivo pickle
model_filename = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\checkpoints\04_lr_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)

print(X.info())