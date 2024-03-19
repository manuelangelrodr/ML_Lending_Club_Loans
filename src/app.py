from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

# Inicializar la aplicación Flask
app = Flask(__name__)

# Define las rutas absolutas para tus datasets
base_dir = r'C:\Users\manue\OneDrive\Documentos\GitHub\ML_Lending_Club_Loans\src'
x_path = os.path.join(base_dir, 'data', 'processed', 'X.csv')
y_path = os.path.join(base_dir, 'data', 'processed', 'y.csv')  # Asume que también tienes y en un archivo separado

# Cargar los datasets
X = pd.read_csv(x_path)
y = pd.read_csv(y_path)

# Definir las columnas y el preprocesador
categorical_cols = ['grade', 'verification_status', 'purpose', 'home_ownership']
numeric_cols = ['loan_amount', 'loan_term', 'interest_rate', 'dti', 'inquiries',
                'derogatory', 'revolving_rate', 'total_credit_lines', 'employment_length', 'annual_income']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


# Configurar y entrenar el modelo de regresión logística
model = LogisticRegression(class_weight='balanced', penalty='l2', C=0.1, solver='saga', max_iter=1000)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X, y)  # Asegúrate de ajustar si 'y' no está en el formato correcto

# Establece el directorio base al directorio donde se encuentra app.py
# base_dir = os.path.dirname(os.path.abspath(__file__))
# preprocessor_path = os.path.join(base_dir, 'checkpoints', '03_preprocessor.pkl')
# model_path = os.path.join(base_dir, 'checkpoints', '04_lr_model.pkl')

# Cargar los archivos pickle necesarios
# with open(preprocessor_path, 'rb') as f:
#     preprocessor = pickle.load(f)
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# Definir la ruta para la página de inicio
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Recopilar datos del formulario
        data = {
            'loan_amount': float(request.form.get('loan_amount')),
            'loan_term': int(request.form.get('loan_term')),
            'interest_rate': float(request.form.get('interest_rate')),
            'grade': request.form.get('grade'),
            'verification_status': request.form.get('verification_status'),
            'purpose': request.form.get('purpose'),
            'dti': float(request.form.get('dti')),
            'inquiries': int(request.form.get('inquiries')),
            'derogatory': int(request.form.get('derogatory')),
            'revolving_rate': float(request.form.get('revolving_rate')),
            'total_credit_lines': int(request.form.get('total_credit_lines')),
            'employment_length': float(request.form.get('employment_length')),
            'home_ownership': request.form.get('home_ownership'),       
            'annual_income': np.log(float(request.form.get('annual_income')) + 1)
        }

        
        # Convertir datos del formulario en DataFrame para el preprocesamiento
        input_data = pd.DataFrame([data], columns=data.keys())

               
        # Preprocesar los datos
        input_data_processed = preprocessor.transform(input_data)
        
        # Hacer la predicción
        prediction = model.predict(input_data_processed)
        result = 'Solicitud de Préstamo Aceptada' if prediction == 0 else 'Revisar Solicitud de Préstamo Con Más Detalle'
        
        # Devolver la plantilla con el resultado de la predicción
        return render_template('index.html', prediction=result)
    else:
        # Devolver la plantilla inicial sin resultado de predicción
        return render_template('index.html', prediction=None)

# Punto de entrada principal
if __name__ == '__main__':
    app.run(debug=True)

