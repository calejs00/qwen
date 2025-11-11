import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from pandas import DataFrame

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
PEFT_ADAPTERS_PATH = "qwen_hora_final"
TEST_DATASET_FILE = "datos_horas_test_random.jsonl"

def load_peft_model_for_inference(model_id, peft_path):
    print("Cargando modelo y tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb_config = torch.quantization.QuantStub()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map=device
    )

    model_with_peft = PeftModel.from_pretrained(base_model, peft_path)
    model_with_peft.eval()
    print(f"Modelo cargado")
    return model_with_peft, tokenizer, device

def generar_hora_digital(instruction: str, modelo, tokenizador):
    prompt = f"{instruction}\nRespuesta:"

    inputs = tokenizador(prompt, return_tensors="pt", padding=True).to(modelo.device)

    with torch.no_grad():
        output_tokens = modelo.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=tokenizador.eos_token_id
        )

    generated_text = tokenizador.decode(output_tokens[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    respuesta_limpia = generated_text.split('\n')[0].strip()
    if "Respuesta:" in respuesta_limpia:
        respuesta_limpia = respuesta_limpia.split("Respuesta:")[1].strip()

    return respuesta_limpia


def evaluate_model():
    from transformers import BitsAndBytesConfig
    model_with_peft, tokenizer, device = load_peft_model_for_inference(MODEL_ID, PEFT_ADAPTERS_PATH)

    try:
        test_dataset = load_dataset("json", data_files=TEST_DATASET_FILE, split="train")
        print(f"Dataset de prueba cargado con {len(test_dataset)} ejemplos.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de prueba: {TEST_DATASET_FILE}. Asegúrate de ejecutar 'generate_test_dataset.py' primero.")
        return

    resultados = []
    correctos = 0
    total = len(test_dataset)

    print("\nIniciando evaluación...")
    for i, ejemplo in enumerate(test_dataset):
        instruction = ejemplo['instruction']
        expected_output = ejemplo['output'].strip()

        predicted_output = generar_hora_digital(instruction, model_with_peft, tokenizer).strip()

        es_correcto = (predicted_output == expected_output)
        if es_correcto:
            correctos += 1

        resultados.append({
            "Entrada": instruction,
            "Esperado": expected_output,
            "Predicho": predicted_output,
            "Correcto": "Correcto" if es_correcto else "Incorrecto"
        })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"Procesado: {i + 1}/{total}")

    accuracy = (correctos / total) * 100

    print("\n" + "="*50)
    print(f"RESULTADOS FINALES EN EL CONJUNTO DE PRUEBA ({total} ejemplos)")
    print("="*50)
    print(f"Total Correctos: {correctos}")
    print(f"Precisión (Accuracy): {accuracy:.2f}%")
    print("="*50)

    df_resultados = DataFrame(resultados)
    df_fallas = df_resultados[df_resultados['Correcto'] == 'Incorrecto']

    if not df_fallas.empty:
        print("\n--- EJEMPLOS FALLIDOS (para análisis) ---")
        df_display = df_fallas[['Entrada', 'Esperado', 'Predicho', 'Correcto']]
        with pd.option_context('display.max_colwidth', None,
                               'display.width', 1000,
                               'display.max_rows', None):

              print(df_display.to_string(index=False))
        print("------------------------------------------")
    else:
        print("\nEl modelo acertó en todos los ejemplos de prueba")


if __name__ == "__main__":
    import pandas as pd
    evaluate_model()