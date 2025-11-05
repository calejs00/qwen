#ejecutado en google colab
# !pip install -U bitsandbytes
# !pip install -U transformers peft accelerate # Asegurar que estas también estén actualizadas
# !pip install fastapi uvicorn python-multipart pyngrok -q

import torch
import os
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

NGROK_TOKEN = "350mZYqwv4HjGKt8RHbDtJAqBxu_59PrgE6xgxN6hQAh8kELP"
ngrok.set_auth_token(NGROK_TOKEN)
PORT = 8000
os.environ['PORT'] = str(PORT)
MERGED_MODEL_DIR = "./qwen_hora_merged_api" #modelo fusionado
print("Cargando modelo fusionado...")

generator = pipeline(
    "text-generation", 
    model=MERGED_MODEL_DIR, 
    torch_dtype=torch.bfloat16 # Ajusta si usaste otro dtype
)
print("Modelo cargado en la GPU.")

from fastapi import FastAPI
from pydantic import BaseModel

class InputData(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.9

app = FastAPI()

@app.post("/generate")
def generate_endpoint(data: InputData):
    formatted_prompt = f"Instrucción: {data.prompt}\nRespuesta: "
    
    params = {
        'max_new_tokens': data.max_new_tokens,
        'temperature': data.temperature,
        'do_sample': True,
        'stop': ["Instrucción:"] # Detener la generación
    }
    
    result = generator(formatted_prompt, **params)
    
    response_text = result[0]["generated_text"].split("Respuesta: ")[-1]
    
    return {"response": response_text.strip()}

# --- INICIAR UVICORN Y NGROK ---
import threading
import uvicorn
from uvicorn.config import Config
from pyngrok import ngrok

PORT = 8000 # Asegúrate de que este es tu puerto configurado

# 1. Crear el túnel público (Ngrok)
http_tunnel = ngrok.connect(PORT)
print(f"URL de API para Bruno (POST): {http_tunnel.public_url}/generate")

# 2. Configurar Uvicorn manualmente
config = Config(
    app=app,             # Tu objeto FastAPI
    host="0.0.0.0",
    port=PORT,
    log_level="warning"  # Silenciar la mayoría de los logs
)
server = uvicorn.Server(config)

# 3. Función para ejecutar el servidor
def run_server():
    # ¡Llamamos al método .run() del objeto Server, no a uvicorn.run()!
    server.run()

# 4. Iniciar el servidor en un hilo separado
thread = threading.Thread(target=run_server)
thread.start()

print("Servidor Uvicorn iniciado en segundo plano. ¡Listo para Bruno!")