from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="API Qwen Extracción de Hora")

class InputData(BaseModel):
    """Estructura de la entrada JSON"""
    texto_hora: str
    api_key: str # Para simular la API Key que validarías (ejemplo)

class OutputData(BaseModel):
    """Estructura de la salida JSON"""
    hora_digital: str
    
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER_DIR = "qwen_hora_final" # La carpeta que creaste en el entrenamiento

MODEL = None
TOKENIZER = None

@app.on_event("startup")
def load_model():
    """Carga el modelo base y los adaptadores QLoRA al iniciar la API."""
    global MODEL, TOKENIZER
    
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16, # Usar el mismo dtype que en el entrenamiento si es posible
        device_map="auto"
    )
    
    MODEL = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    MODEL.eval()
    print("Modelo Qwen con LoRA cargado y listo.")


@app.post("/extract_time", response_model=OutputData)
async def extract_time_api(data: InputData):
    """Endpoint principal para la conversión de texto a hora digital."""
    
    if data.api_key != "TU_CLAVE_SECRETA":
         raise HTTPException(status_code=401, detail="API Key no válida")
    
    instruction = f"Convierte esta hora de texto a formato digital de 24 horas: {data.texto_hora}"
    
    inputs = TOKENIZER(instruction, return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=10, # Solo necesitamos unos pocos tokens (la hora)
            do_sample=False,
            temperature=0.0 # Usar 0.0 para que la salida sea determinista (no creativa)
        )
    
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    hora_digital = response.split("24 horas:")[-1].strip()
    
    
    return OutputData(hora_digital=hora_digital)

