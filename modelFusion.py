#para transformar los componentes del entrenamiento en un único archivo para utilizar en API
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FINAL_MODEL_DIR = "qwen_hora_final" 
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat" 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Cargando modelo base con cuantización a 4-bit...")
model_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config, 
    device_map="cpu",              
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token 

print("Cargando adaptadores LoRA...")
model = PeftModel.from_pretrained(
    model_base,
    FINAL_MODEL_DIR
)

# FUSIONAR los pesos LoRA
print("Fusionando adaptadores LoRA en el modelo base...")
merged_model = model.merge_and_unload()

MERGED_MODEL_DIR = "qwen_hora_merged_api" 
merged_model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"Modelo fusionado listo en: {MERGED_MODEL_DIR}")