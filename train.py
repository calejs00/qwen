import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DATASET_FILE = "datos_horas.jsonl"
OUTPUT_DIR = "qwen_hora_qlora"

dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=32,                  # Rango de LoRA (tamaño de los adaptadores)
    lora_alpha=64,         # Escalado
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Las capas a modificar
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False # Desactivar caché para el entrenamiento

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                     # Número de veces que el modelo ve todos los datos
    per_device_train_batch_size=4,          # Ejemplos por lote (ajustar según tu VRAM)
    gradient_accumulation_steps=4,          # Acumulación de gradientes (simula un batch size más grande)
    optim="paged_adamw_8bit",               # Optimizador eficiente para QLoRA
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True, # Si usas PyTorch reciente con una GPU moderna, pon esto a True
)

def formatting_function(example):
    """
    Combina las columnas 'instruction' y 'output' en un único texto
    para que el modelo aprenda el patrón.
    """
    # El SFTTrainer pasa UN SOLO diccionario de ejemplo a la vez
    instruction = example['instruction']
    output = example['output']
    
    # Este es el formato de instrucción que el modelo verá:
    # Usamos una estructura que Qwen entiende bien para tareas de Q&A:
    text = f"Instrucción: {instruction}\nRespuesta: {output}"
    
    return text

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    # tokenizer=tokenizer,
    # packing=False,
    formatting_func=formatting_function,
)

print("Iniciando entrenamiento...")
trainer.train()

final_model_dir = "qwen_hora_final"
trainer.model.save_pretrained(final_model_dir)

print(f"Entrenamiento completado. Pesos guardados en: {final_model_dir}")