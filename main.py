from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import timedelta


# Cargar el modelo
modelo = joblib.load("modelo_entrenado.pkl")

app = FastAPI()
application = app  # <- Esto es importante para Beanstalk

# Este es el punto de entrada WSGI que necesita EB

class DatosEntrada(BaseModel):
    fecha_facturacion: str  # formato YYYY-MM-DD
    monto: float
    dias_factura_a_maximo: int

@app.post("/recomendar")
def recomendar(data: DatosEntrada):
    try:
        fecha = pd.to_datetime(data.fecha_facturacion)
        entrada = pd.DataFrame([{
            "monto": data.monto,
            "dias_factura_a_maximo": data.dias_factura_a_maximo,
            "dia_semana_factura": fecha.weekday(),
            "mes_factura": fecha.month
        }])

        dias_estimados = modelo.predict(entrada)[0]
        fecha_pago_estimada = fecha + timedelta(days=round(dias_estimados))
        return {
            "fecha_estimada_pago": fecha_pago_estimada.strftime("%Y-%m-%d"),
            "dia_recomendado_para_cobro": (fecha_pago_estimada - timedelta(days=2)).strftime("%Y-%m-%d")
        }
    except Exception as e:
        return {"error": str(e)}
