import pandas as pd
import numpy as np
import warnings
import random
import seaborn as sns
# Statistics & Mathematics
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import shapiro, skew, anderson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import math

# Desactiva temporalmente las advertencias de tipo UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

datos= pd.read_csv('C:/Users/luasp/OneDrive/Documents/Visual Studio Code/Fund_Analit/src/credit_risk_dataset.csv', sep=',')
datos.rename(columns={"loan_status ":'Y',
                     "person_age ":'X1', ##
                     "person_income ":'X2',
                     "person_home_ownership ":'X3',
                     "person_emp_length ":'X4',##
                     "loan_intent       ":'X5',
                     "loan_grade ":'X6',
                     "loan_amnt ":'X7',
                     "loan_int_rate ":'X8',
                     "loan_percent_income ":'X9',
                     "cb_person_default_on_file ":'X10',
                     "cb_person_cred_hist_length":'X11'##
                     },
               inplace=True)
#peso de cada variable
datos = datos[['Y','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]

# Crear un nuevo DataFrame con los valores únicos por columna
valores_unicos_por_columna = pd.DataFrame({
    'Columna': datos.columns,
    'Valores Únicos': [datos[col].unique() for col in datos.columns]
})

cant_v_unic = datos.nunique()
cant_v_unic = cant_v_unic.sort_values(ascending=False)

#Eliminamos los registros donde hayan datos vacíos
datos['X8'] = datos['X8'].replace('              ', 0, regex=True)
datos=datos.dropna(subset=['X8'])
datos['X8'] = datos['X8'].astype(float)
print("DATOS:",datos['X8'])
print("DATOS:",datos.dtypes['X8'])
#Eliminar duplicados
datos = datos.drop_duplicates()

##CORRECCIONES DE VARIABLES

#Corrección variable X1
# Calcular la mediana de la columna excluyendo los valores datos cambiar
medianaX1 = datos[datos['X1'].isin([123, 144]) == False]['X1'].median()
# Reemplazar los valores
datos['X1'] = datos['X1'].replace([123, 144], medianaX1)

#Eliminacion de variables poco relevantes
data_=datos.drop(columns=['X1','X11','X4'])

#Transformación de variables
data_['X10'] = data_['X10'].replace({'Y': 1, 'N': 0})
data_['X10'].value_counts().reset_index()

#Hacemos una copia del dataset para aplicar el modelo
dataEnd1 = data_.copy()

# Aplicar codificación one-hot datos la columna 'X3' 'X5' y 'X6'
dataEnd1= pd.get_dummies(dataEnd1,columns=['X6'],prefix=['X6'])
dataEnd1 = pd.get_dummies(dataEnd1, columns=['X3'], prefix=['X3'])
dataEnd1 = pd.get_dummies(dataEnd1, columns=['X5'], prefix=['X5'])

def generar_intervalos(dataset,feature,num_cuartiles):
  bins=[dataset[feature].min()] #Añadimos el valor minimo, para que los contenedores empiecen desde ahí
  cuartiles = [dataset[feature].quantile(q) for q in [i / num_cuartiles for i in range(1, num_cuartiles)]]
  for cuartil in cuartiles: #Añadimos los cuartiles
    bins.append(cuartil)
  bins.append(float('inf'))
  return bins

def convertir_a_dummy(df, columnas, intervalos):
    df_nuevo = df.copy()

    for col, bins in zip(columnas, intervalos):
        intervalos_cortados = pd.cut(df_nuevo[col], bins=bins, right=False)
        variables_dummy = pd.get_dummies(intervalos_cortados, prefix=f'{col}')
        df_nuevo.drop(columns=[col], inplace=True)
        df_nuevo = pd.concat([df_nuevo, variables_dummy], axis=1)
    return df_nuevo

intervalos_x2=generar_intervalos(dataEnd1,'X2',20)
intervalos_x7=generar_intervalos(dataEnd1,'X7',16)
intervalos_x8=generar_intervalos(dataEnd1,'X8',20)
intervalos_x9=generar_intervalos(dataEnd1,'X9',20)

#dataEnd1 = convertir_a_dummy(dataEnd1,['X2','X7','X8','X9'],[intervalos_x2,intervalos_x7,intervalos_x8,intervalos_x9])

# #Ajuste del modelo
# #Dividimos los datos que vamos datos usar para ajustar el modelo de Regresión Logistica
# datos_tr_log, datos_vl_log = train_test_split(dataEnd1, train_size=0.8, random_state=20)
# # Extraer la columna Y de los conjuntos de entrenamiento y validación
# Y_tr_log = datos_tr_log["Y"]
# Y_vl_log = datos_vl_log["Y"]
# # Extraer todas las columnas excepto Y de los conjuntos de entrenamiento y validación
# X_tr_log = datos_tr_log.drop(columns=["Y"])
# X_vl_log = datos_vl_log.drop(columns=["Y"])

# # Inicialización del modelo
# mod1 = LogisticRegression(max_iter=1000, class_weight = 'balanced')
# # Entrenamiento
# mod1.fit(X_tr_log, Y_tr_log)

# # Realizar predicciones en el conjunto de validación
# Y_pred_mod1 = mod1.predict(X_vl_log)
# accuracy_mod1 = metrics.accuracy_score(Y_vl_log, Y_pred_mod1)
# precision_mod1 = metrics.precision_score(Y_vl_log, Y_pred_mod1)
# recall_mod1 = metrics.recall_score(Y_vl_log, Y_pred_mod1)
# f1_mod1 = metrics.f1_score(Y_vl_log, Y_pred_mod1)

# #Probabilidades de predicción en el conjunto de validación
# prob_predict_mod1 = mod1.predict_proba(X_vl_log)[:, 1]
# fpr1, tpr1, umbrales1 = metrics.roc_curve(Y_vl_log, prob_predict_mod1)

# # Calcular el área bajo la curva ROC (AUC)
# auc_mod1 = metrics.roc_auc_score(Y_vl_log, prob_predict_mod1)

# # Resumen de las métricas obtenidas
# modelo1_metrics = {'Accuracy': accuracy_mod1, 'Precisión': precision_mod1, 'Recall': recall_mod1, 'AUC': auc_mod1}

# #Score card modelo 1 - cuadro de mandos

# coeficientes = mod1.coef_
# intercepto = mod1.intercept_
# # Define the min and max threshholds for our scorecard
# min_score = 300
# max_score = 850
# # Crear un DataFrame con los coeficientes y el intercepto
# ## IMPORTANTE:  Se multiplican los coeficientes por -1, ya que el valor positivo es que INCUMPLIO, pero nosotros datos la persona que CUMPLE es la que le queremos dar más score.
# coeficientes_dict = {'Feature name': ['Intercept'] + list(X_tr_log.columns), 'Coefficients': [intercepto[0]] + list(coeficientes[0]*-1)}
# df_scorecard = pd.DataFrame(coeficientes_dict)
# #Nombre original de la variable
# df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split('_').str[0]
# # calculate the sum of the minimum coefficients of each category within the original feature name
# min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
# # calculate the sum of the maximum coefficients of each category within the original feature name
# max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
# # create datos new columns that has the imputed calculated Score based on the multiplication of the coefficient by the ratio of the differences between
# # maximum & minimum score and maximum & minimum sum of cefficients.
# df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
# # update the calculated score of the Intercept (i.e. the default score for each loan)
# df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0,'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
# # round the values of the 'Score - Calculation' column and store them in datos new column
# df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
# # check the min and max possible scores of our scorecard
# min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
# max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
# #print(min_sum_score_prel)
# #print(max_sum_score_prel)
# df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
# df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']

# #Obtener el score con la formula
# X_vl_log_score = X_vl_log.copy()
# X_vl_log_score.insert(0, 'Intercept', 1)

# # Obtenemos lo valores asociados datos cada variable, del score
# scorecard_scores = df_scorecard.set_index('Feature name')['Score - Final']
# #Hacemos la multiplicación del score por cada fila de nuestro dataframe de los datos de validación.
# X_vl_log_score['Score'] = X_vl_log_score.apply(lambda row: sum(row * scorecard_scores), axis=1)

# # Crear un DataFrame con los valores reales y predichos
# probabilidades_prediccion_mod1 = mod1.predict_proba(X_vl_log)  # Obtener las probabilidades
# df_predicciones_mod1 = pd.DataFrame({'loan_status_Real': Y_vl_log, 'loan_status_predict': Y_pred_mod1, 'probability_of_good_payer': probabilidades_prediccion_mod1[:,0]})
# df_predicciones_mod1['Score final']=X_vl_log_score['Score']

# #print(valores_unicos_por_columna)