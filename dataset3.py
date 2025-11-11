import json
import random
from datetime import datetime, timedelta

# --- Constantes y Variaciones Lingüísticas ---
PREPOSITIONS_Y = ["y", "con", "pasadas las"]
PREPOSITIONS_MENOS = ["menos", "para las"]
UNIDADES = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
PREFIXES = [
    "convierte esta hora en texto: ",
    "escribe la hora en palabras: ",
    "la hora es: ",
    "ahora mismo son las: ",
    "dime la hora: ",
    "transforma a formato 24 horas: ", # Nuevo prefijo
]
TIMES_OF_DAY = {
    (0, 5): ["de la madrugada", "a.m."],
    (6, 11): ["de la mañana", "a.m."],
    (12, 16): ["del mediodía", "de la tarde", "p.m."],
    (17, 23): ["de la tarde", "de la noche", "p.m."],
}

# La función number_to_spanish debe ser robusta, la mantengo igual que en tu base
def number_to_spanish(n):
    """Convierte números del 1 al 59 a texto"""
    if not (0 <= n <= 59): return str(n)
    unidades = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    dieces = ["diez", "once", "doce", "trece", "catorce", "quince", "dieci", "veinte", "treinta", "cuarenta", "cincuenta"]

    if n == 0: return "cero"
    if n < 10: return unidades[n]
    if n <= 15: return dieces[n-10]
    if n < 20: return "dieci" + unidades[n % 10]
    if n == 20: return "veinte"
    if n < 30:
        if n == 20: return "veinte"
        return "veinti" + unidades[n % 10]
    
    decena_index = int(n / 10) - 1
    decena = dieces[decena_index + 5]
    unidad = unidades[n % 10]
    
    if n % 10 == 0: return decena
    else: return f"{decena} y {unidad}"
    
def get_time_of_day(h_24):
    """Devuelve el momento del día."""
    for (start, end), choices in TIMES_OF_DAY.items():
        if start <= h_24 <= end:
            return random.choice(choices)
    return random.choice(["de la noche", "a.m.", "de la madrugada"])


def convert_to_text(h_24, m):
    """Genera una lista de frases de hora para HH:MM con múltiples variaciones."""
    text_forms = []
    
    h_12 = h_24 % 12
    h_12 = 12 if h_12 == 0 else h_12
    h_text = number_to_spanish(h_12)
    momento = get_time_of_day(h_24)

    # 1. Formato 'y [minutos]' (00:01 a 00:59)
    if m != 0:
        m_text = number_to_spanish(m)
        text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} {m_text} {momento}")
        # Variación con 'minutos' explícito para minutos de un dígito
        if m < 10:
             text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} cero {m_text} minutos {momento}")

    # 2. Formato 'menos [minutos]' (00:31 a 00:59)
    if m > 30:
        m_rest = 60 - m
        m_rest_text = number_to_spanish(m_rest)
        h_next_24 = (h_24 + 1) % 24
        h_next_12 = h_next_24 % 12
        h_next_12 = 12 if h_next_12 == 0 else h_next_12
        h_next_text = number_to_spanish(h_next_12)
        momento_next = get_time_of_day(h_next_24)
        
        text_forms.append(f"las {h_next_text} {random.choice(PREPOSITIONS_MENOS)} {m_rest_text} {momento_next}")

    # 3. Formatos especiales (cuarto, media, en punto)
    if m == 0:
        text_forms.append(f"las {h_text} en punto {momento}")
        text_forms.append(f"las {h_text} {momento}")
    elif m == 30:
        text_forms.append(f"las {h_text} y media {momento}")
    elif m == 15:
        text_forms.append(f"las {h_text} y cuarto {momento}")
    elif m == 45:
        h_next_24 = (h_24 + 1) % 24
        h_next_12 = h_next_24 % 12
        h_next_12 = 12 if h_next_12 == 0 else h_next_12
        h_next_text = number_to_spanish(h_next_12)
        momento_next = get_time_of_day(h_next_24)
        text_forms.append(f"las {h_next_text} menos cuarto {momento_next}")

    # 4. Formato 24h explícito (para ayudar a la conversión)
    text_forms.append(f"{number_to_spanish(h_24)} horas y {number_to_spanish(m)} minutos")
    
    return text_forms

def generate_dataset_v3(num_iterations=3, output_file='datos_horas_v3.jsonl'):
    dataset = []
    base_time = datetime(2025, 1, 1)
    
    for iteration in range(num_iterations):
        # Iterar a través de CADA MINUTO del día (1440 minutos)
        for i in range(24 * 60):
            current_time = base_time + timedelta(minutes=i)
            h_24 = current_time.hour
            m = current_time.minute
            
            salida_digital = f"{h_24:02d}:{m:02d}"

            input_texts = convert_to_text(h_24, m)
            
            for text in input_texts:
                full_instruction = random.choice(PREFIXES) + text
                
                dataset.append({
                    "instruction": full_instruction.strip(),
                    "output": salida_digital
                })

    # Añadir casos especiales
    dataset.append({"instruction": random.choice(PREFIXES) + "medianoche", "output": "00:00"})
    dataset.append({"instruction": random.choice(PREFIXES) + "el mediodía", "output": "12:00"})
    dataset.append({"instruction": random.choice(PREFIXES) + "doce de la noche", "output": "00:00"})
    
    random.shuffle(dataset) # Barajar el dataset
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dataset generado: {output_file} con {len(dataset)} ejemplos.")
    return output_file

if __name__ == "__main__":
    generate_dataset_v3(num_iterations=3) # Genera 3 ciclos completos del día