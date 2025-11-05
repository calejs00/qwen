import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER_DIR = "qwen_hora_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("Modelo entrenado cargado. Realizando prueba...")

def generar_hora(texto_entrada):
    prompt = f"Instrucción: Convierte esta hora en texto a formato digital de 24 horas: {texto_entrada}\nRespuesta:"

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inputs = inputs.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        resultado = response.split("Respuesta:")[-1].strip()
    except:
        resultado = "Error al limpiar la respuesta."

    return resultado

# --- PRUEBAS ---
prueba_1 = "las seis y media de la tarde"
prueba_2 = "dos de la mañana"
prueba_3 = "una de la mañana" #este lo hace mal, pone 00:00

print(f"\nEntrada 1: {prueba_1}")
print(f"Salida Modelo: {generar_hora(prueba_1)}")

print(f"\nEntrada 2: {prueba_2}")
print(f"Salida Modelo: {generar_hora(prueba_2)}")

print(f"\nEntrada 3: {prueba_3}")
print(f"Salida Modelo: {generar_hora(prueba_3)}")