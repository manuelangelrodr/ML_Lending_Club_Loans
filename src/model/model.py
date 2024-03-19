# IMPORTAR LIBRERIAS
import numpy as np
import pandas as pd
import os

file_path = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src\data\processed\df_original.csv'

# import sys
# sys.path.append('../')
# from utils.eda_functions import *


# Cargar el archivo CSV en un DataFrame de pandas
df = pd.read_csv(file_path)


print(df.head())
