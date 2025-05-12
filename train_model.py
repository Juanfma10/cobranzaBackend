import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# Cargar datos
df = pd.read_csv("registros_de_pago.csv", parse_dates=["dia_facturacion", "dia_pago", "dia_maximo_pago"])

# Preprocesamiento
df['dias_para_pagar'] = (df['dia_pago'] - df['dia_facturacion']).dt.days
df['dias_factura_a_maximo'] = (df['dia_maximo_pago'] - df['dia_facturacion']).dt.days
df['dia_semana_factura'] = df['dia_facturacion'].dt.weekday
df['mes_factura'] = df['dia_facturacion'].dt.month

df = df.dropna()

X = df[['monto', 'dias_factura_a_maximo', 'dia_semana_factura', 'mes_factura']]
y = df['dias_para_pagar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Modelos a evaluar
modelos = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

try:
    from xgboost import XGBRegressor
    modelos["XGBoost"] = XGBRegressor(random_state=42)
except ImportError:
    print("XGBoost no está instalado, se omitirá.")

# Entrenar y seleccionar mejor modelo
mejor_mae = float("inf")
mejor_modelo = None
mejor_nombre = ""

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"{nombre} -> MAE: {mae:.2f}")
    if mae < mejor_mae:
        mejor_mae = mae
        mejor_modelo = modelo
        mejor_nombre = nombre

print(f"\n✅ Mejor modelo: {mejor_nombre} (MAE: {mejor_mae:.2f})")

# Guardar el mejor modelo
joblib.dump(mejor_modelo, "modelo_entrenado.pkl")
