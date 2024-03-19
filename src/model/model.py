# IMPORTAR LIBRERIAS

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_score
import pickle

import sys
sys.path.append('../')
from utils.eda_functions import *

# CARGAMOS EL DATASET

df = pd.read_csv('../data/processed/df_original.csv')

print(df.head(10))