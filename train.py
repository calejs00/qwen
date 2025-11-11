import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DATASET_FILE = "datos_horas.jsonl"
OUTPUT_DIR = "qwen_hora_qlora"

def formatting_function(example):
    instruction = example['instruction']
    output = example['output']

    text = f"Instrucción: {instruction}\nRespuesta: {output}"

    return text

def trainModel():
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
    )


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

    trainer.model.save_pretrained("qwen_hora_final")
    print("Entrenamiento completado. Pesos guardados en: qwen_hora_final")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype = torch.bfloat16)
    modelPert = PeftModel.from_pretrained(base_model, "qwen_hora_final")
    merged_model = modelPert.merge_and_unload()
    merged_model.save_pretrained("qwen_hora_fusionado")
    tokenizer.save_pretrained("qwen_hora_fusionado")

    print("Modelo fusionado guardado")

if __name__ == "__main__":
    trainModel()