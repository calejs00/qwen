import json
import random
from datetime import datetime, timedelta

# --- CONSTANTES ---
DIAS_SEMANA_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

EXPRESIONES_RELATIVAS = {
    "una hora": timedelta(hours=1),
    "dos horas": timedelta(hours=2),
    "una hora y media": timedelta(hours=1, minutes=30),
    "veinte minutos": timedelta(minutes=20),
    "treinta minutos": timedelta(minutes=30),
}

EXPRESIONES_PARTE_DIA = {
    "esta noche": lambda t: t.replace(hour=21, minute=0, second=0) if t.hour < 20 else t + timedelta(days=1, hours=21),
    "esta tarde": lambda t: t.replace(hour=18, minute=0, second=0) if t.hour < 17 else t + timedelta(days=1, hours=18),
    "esta mañana": lambda t: t.replace(hour=9, minute=0, second=0) if t.hour < 8 else t + timedelta(days=1, hours=9),
}

# --- FUNCIONES DE CONVERSIÓN DE HORA A TEXTO ---

def number_to_spanish(n):
    """Convierte números del 1 al 59 a texto, y horas clave."""
    if n == 0: return "doce" # Para 12 am/pm
    
    unidades = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    dieces = ["diez", "once", "doce", "trece", "catorce", "quince"]
    
    if 1 <= n <= 9: return unidades[n]
    if 10 <= n <= 15: return dieces[n-10]
    if 16 <= n <= 19: return "dieci" + unidades[n % 10]
    if n == 20: return "veinte"
    if 21 <= n <= 29: return "veinti" + unidades[n % 10]
    
    if n < 60:
        decena = ["", "", "", "treinta", "cuarenta", "cincuenta"][n // 10]
        unidad = unidades[n % 10]
        return f"{decena} y {unidad}" if n % 10 != 0 else decena
    return str(n)


def convert_to_text_h_m(h_24, m):
    """Genera una hora en texto y su descriptor (mañana/tarde/noche)."""
    h_12 = h_24 % 12
    h_12 = 12 if h_12 == 0 else h_12 # 12 para 00:00 y 12:00
    h_text = number_to_spanish(h_12)
    
    if 6 <= h_24 < 12:
        momento = "de la mañana"
    elif 12 <= h_24 < 20:
        momento = "de la tarde"
    else:
        momento = "de la noche"
        
    m_text = number_to_spanish(m)
    
    # Priorizar frases cortas para peticiones (y cuarto, y media, menos cuarto)
    if m == 0:
        return f"{h_text} en punto {momento}"
    elif m == 30:
        return f"{h_text} y media {momento}"
    elif m == 15:
        return f"{h_text} y cuarto {momento}"
    elif m == 45:
        h_next_12 = (h_24 + 1) % 12
        h_next_12 = 12 if h_next_12 == 0 else h_next_12
        h_next_text = number_to_spanish(h_next_12)
        return f"{h_next_text} menos cuarto {momento}"
    else:
        return f"{h_text} y {m_text} {momento}"


def generate_tlp_dataset(num_examples=12000, output_file='datos_horas_tlp_v4.jsonl'):
    dataset = []
    
    # Usaremos 5000 iteraciones y crearemos 2-3 ejemplos por iteración
    for _ in range(num_examples // 2):
        
        # 1. Generar un Contexto Base Aleatorio (el 'AHORA')
        base_timestamp = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365), 
                                                           hours=random.randint(0, 23), 
                                                           minutes=random.randint(0, 59))
        
        contexto_base_str = base_timestamp.strftime("%Y-%m-%d %H:%M")
        
        
        # --- CASO 1: Referencia Relativa ("dentro de X tiempo") ---
        expresion, delta = random.choice(list(EXPRESIONES_RELATIVAS.items()))
        peticion_relativa = f"Quiero un taxi **dentro de {expresion}**."
        salida_relativa = (base_timestamp + delta).strftime("%Y-%m-%d %H:%M")
        dataset.append({"peticion": peticion_relativa, 
                        "contexto_base": contexto_base_str, 
                        "salida_absoluta": salida_relativa})
        
        # --- CASO 2: Día Específico + HORA EN TEXTO (Tu objetivo) ---
        
        # Generar una hora y minuto aleatorios (no en el contexto_base)
        target_timestamp = base_timestamp + timedelta(days=random.randint(1, 7), 
                                                      hours=random.randint(-12, 12), 
                                                      minutes=random.randint(-30, 30))
        
        target_h_24 = target_timestamp.hour
        target_m = target_timestamp.minute
        
        # 2a. Convertir HORA y MINUTO a TEXTO
        hora_texto = convert_to_text_h_m(target_h_24, target_m)
        hora_abs = f"{target_h_24:02d}:{target_m:02d}"
        
        # 2b. Seleccionar un modificador de día
        opciones_dia = ["mañana", "pasado mañana", f"el {random.choice(DIAS_SEMANA_ES)}"]
        modificador_dia = random.choice(opciones_dia)
        
        peticion_abs = f"Quiero reservar un taxi para **{modificador_dia} a las {hora_texto}**."

        # Calcular la fecha absoluta (simplificado, solo para generar el output correcto)
        if "mañana" in modificador_dia:
            delta_days = 1
        elif "pasado mañana" in modificador_dia:
            delta_days = 2
        else:
            dia_buscado = modificador_dia.split()[1] 
            dia_actual = base_timestamp.weekday()
            dia_objetivo = DIAS_SEMANA_ES.index(dia_buscado) 
            delta_days = (dia_objetivo - dia_actual) % 7
            if delta_days == 0: delta_days += 7

        fecha_abs = (base_timestamp + timedelta(days=delta_days)).strftime("%Y-%m-%d")
        salida_abs = f"{fecha_abs} {hora_abs}"
        
        dataset.append({"peticion": peticion_abs, 
                        "contexto_base": contexto_base_str, 
                        "salida_absoluta": salida_abs})

        # --- CASO 3: Referencia de Parte del Día ("Esta Noche") ---
        if random.random() < 0.5: 
             expresion, func = random.choice(list(EXPRESIONES_PARTE_DIA.items()))
             hora_recogida = func(base_timestamp)
             peticion_parte = f"Necesito un taxi para **{expresion}**."
             salida_parte = hora_recogida.strftime("%Y-%m-%d %H:%M")
             
             dataset.append({"peticion": peticion_parte, 
                             "contexto_base": contexto_base_str, 
                             "salida_absoluta": salida_parte})


    random.shuffle(dataset) 
    
    # Asegurar al menos 10.000 ejemplos
    dataset = dataset[:num_examples] if len(dataset) > num_examples else dataset

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Dataset TLP V5 generado: {output_file} con {len(dataset)} ejemplos.")

if __name__ == "__main__":
    generate_tlp_dataset(num_examples=12000)