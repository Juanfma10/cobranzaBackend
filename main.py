from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"mensaje": "¡Hola desde FastAPI!"}

@app.get("/saludo/{nombre}")
async def saludar(nombre: str):
    return {"saludo": f"Hola, {nombre}!"}
