import json
import random
from datetime import datetime, timedelta

PREPOSITIONS_Y = ["y", "con"]
PREPOSITIONS_MENOS = ["menos"]
PREFIXES = [
    "convierte esta hora en texto: ",
    "escribe la hora en palabras: ",
    "¿cómo se dice esta hora?: ",
    "ahora mismo son las: ",
]

def number_to_spanish(n):
    """convierte números del 1 al 59 a texto"""
    if not (0 <= n <= 59):
        return str(n)
    
    unidades = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    dieces = ["diez", "once", "doce", "trece", "catorce", "quince", "dieci", "veinte", "treinta", "cuarenta", "cincuenta"]

    if n == 0:
        return "cero"
    if n < 10:
        return unidades[n]    
    if n <= 15:
        return dieces[n-10] 
    if n < 20:
        return "dieci" + unidades[n % 10]
    if n == 20:
        return "veinte"
    if n < 30:
        if n == 20: return "veinte"
        return "veinti" + unidades[n % 10]
    
    decena_index = int(n / 10) + 5
    decena = dieces[decena_index]
    unidad = unidades[n % 10]
    
    if n % 10 == 0:
        return decena
    else:
        return f"{decena} y {unidad}"
    
def get_time_of_day(h_24):
    """Devuelve 'de la mañana', 'de la tarde', 'de la noche'"""
    if 6 <= h_24 < 12:
        return random.choice(["de la mañana", "a.m."])
    elif 12 <= h_24 < 20:
        return random.choice(["de la tarde", "p.m.", "del mediodía"])
    else:
        return random.choice(["de la noche", "a.m."])
    
def convert_to_text(h_24, m):
    """Genera una frase de hora en español para HH:MM"""
    
    text_forms = []
    
    # --- Formato 1: Horario 12h (Mañana/Tarde/Noche) ---
    h_12 = h_24 % 12
    h_12 = 12 if h_12 == 0 else h_12

    # Horas
    h_text = number_to_spanish(h_12)
    momento = get_time_of_day(h_24)

    # Minutos clave (y media, y cuarto, menos cuarto, en punto)
    if m == 0:
        text_forms.append(f"las {h_text} en punto {momento}")
    elif m == 30:
        text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} media {momento}")
    elif m == 15:
        text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} cuarto {momento}")
    elif m == 45:
        # Hora siguiente para "menos cuarto"
        h_next_12 = (h_24 + 1) % 12
        h_next_12 = 12 if h_next_12 == 0 else h_next_12
        h_next_text = number_to_spanish(h_next_12)
        text_forms.append(f"las {h_next_text} menos cuarto {momento}")
    elif 1 < m < 30:
        # Minutos simples 'y [minutos]'
        m_text = number_to_spanish(m)
        text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} {m_text} {momento}")
    elif 30 < m < 60:
        # Minutos 'menos [minutos]'
        m_rest = 60 - m
        m_rest_text = number_to_spanish(m_rest)
        h_next_12 = (h_24 + 1) % 12
        h_next_12 = 12 if h_next_12 == 0 else h_next_12
        h_next_text = number_to_spanish(h_next_12)
        text_forms.append(f"las {h_next_text} {random.choice(PREPOSITIONS_MENOS)} {m_rest_text} {momento}")

    # --- Formato 2: Horas y Minutos Numéricos (Más fácil para el modelo) ---
    m_text_num = number_to_spanish(m)
    text_forms.append(f"las {h_text} {random.choice(PREPOSITIONS_Y)} {m_text_num} {momento}")

    # --- Formato 3: Horario 24h Explícito ---
    h_24_text = number_to_spanish(h_24) if h_24 != 0 else "cero"
    text_forms.append(f"{h_24_text} horas y {m_text_num} minutos")
    
    return text_forms

def generate_dataset(num_examples=5000):
    dataset = []
    
    # Usaremos una base de tiempo para generar variaciones consistentes
    base_time = datetime(2025, 1, 1)
    
    # Definir pasos para asegurar una distribución uniforme de minutos (cada 5 minutos)
    time_steps = [base_time + timedelta(minutes=i) for i in range(0, 24 * 60, 5)]

    # Asegurar que generamos el número de ejemplos deseado
    while len(dataset) < num_examples:
        
        # 1. Seleccionar un punto de tiempo aleatorio de nuestros pasos
        current_time = random.choice(time_steps)
        h_24 = current_time.hour
        m = current_time.minute
        
        # 2. Generar la salida (el objetivo del modelo)
        salida_digital = f"{h_24:02d}:{m:02d}"

        # 3. Generar la(s) entrada(s) de texto
        input_texts = convert_to_text(h_24, m)
        
        # 4. Crear un registro por cada variante de texto
        for text in input_texts:
            if len(dataset) < num_examples:
                # Añadir un prefijo de instrucción aleatorio
                full_instruction = random.choice(PREFIXES) + text
                
                dataset.append({
                    "instruction": full_instruction,
                    "output": salida_digital
                })

    # Añadir casos especiales
    dataset.append({"instruction": random.choice(PREFIXES) + "medianoche", "output": "00:00"})
    dataset.append({"instruction": random.choice(PREFIXES) + "el mediodía", "output": "12:00"})
    dataset.append({"instruction": random.choice(PREFIXES) + "mediodía en punto", "output": "12:00"})
    dataset.append({"instruction": random.choice(PREFIXES) + "doce de la noche", "output": "00:00"})
    
    # Eliminar duplicados si los hay (aunque la generación varía por el prefijo)
    # Convertir a JSONL
    with open('datos_horas.jsonl', 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dataset generado: datos_horas.jsonl con {len(dataset)} ejemplos.")

if __name__ == "__main__":
    generate_dataset(num_examples=10000) # Generaremos 10,000 ejemplos para una buena robustez