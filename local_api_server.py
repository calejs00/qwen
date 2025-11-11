# --- Parte 1: Importaciones y Configuración ---
import torch
import re
import pandas as pd
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset # Necesario para cargar las clases de datasets
from peft import PeftModel # Necesario para la clase PeftModel

# Configuración del servidor
app = Flask(__name__)
PORT = 5000

# Rutas del modelo
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
FUSED_MODEL_PATH = "qwen_hora_fusionado" # Tu modelo final fusionado (TLP)

# Variables globales para el modelo
model = None
tokenizer = None
device = None

# --- Parte 2: Lógica de Carga y Predicción del Modelo ---

def load_model():
    """Carga el modelo fusionado en el dispositivo disponible (CPU o CUDA)."""
    global model, tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo TLP en: {device}")
    
    # El modelo fusionado se carga directamente, sin PEFT ni BNB (asumiendo que está fusionado)
    model = AutoModelForCausalLM.from_pretrained(
        FUSED_MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("Modelo Qwen TLP cargado exitosamente.")

def predict_time(peticion: str, contexto_base: str):
    """Genera la hora absoluta a partir de la petición y el contexto base."""
    if model is None or tokenizer is None:
        return "Error: Modelo no cargado", 500

    # 1. Crear el prompt en el formato TLP del entrenamiento
    prompt = (
        f"Contexto_AHORA: {contexto_base}\n"
        f"Peticion_Usuario: {peticion}\n"
        f"Salida_ABSOLUTA:"
    )

    # 2. Tokenización
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # 3. Generación
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=25, # Suficiente para YYYY-MM-DD HH:MM
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. Decodificación y post-procesamiento
    generated_text = tokenizer.decode(output_tokens[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 5. Extracción robusta del formato TLP (YYYY-MM-DD HH:MM)
    # Busca la salida de la reserva (ej: 2025-11-12 19:30)
    match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', generated_text)
    
    if match:
        return match.group(0)
    else:
        # En caso de que el modelo falle, devolver la salida completa para debug
        return f"ERROR_PARSING: {generated_text.strip()}"
    
    # --- Parte 3: El Endpoint de la API y Ejecución ---

@app.route('/predict_time', methods=['POST'])
def predict():
    """Endpoint para recibir la petición JSON y devolver la hora absoluta."""
    try:
        data = request.get_json()
        peticion = data.get('peticion')
        contexto_base = data.get('contexto_base')

        if not peticion or not contexto_base:
            return jsonify({"error": "Faltan los campos 'peticion' o 'contexto_base' en el JSON."}), 400

        salida_absoluta = predict_time(peticion, contexto_base)
        
        # Devolver el resultado
        return jsonify({
            "peticion_recibida": peticion,
            "contexto_base": contexto_base,
            "salida_absoluta": salida_absoluta
        })

    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    # Cargar el modelo ANTES de arrancar el servidor
    load_model()
    print("\nServidor listo para recibir peticiones POST en /predict_time.")
    app.run(host='0.0.0.0', port=PORT, debug=False)