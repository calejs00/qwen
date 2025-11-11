import json
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)
PORT = 5000 

def mock_predict_time(peticion: str, contexto_base_str: str) -> str:
    """
    Simula la predicción del modelo TLP (Temporal Language Processing).
    Reemplaza la lógica de inferencia del LLM.
    """
    
    # Convierte el contexto base (hora actual) en un objeto datetime
    try:
        # Se asume el formato YYYY-MM-DD HH:MM
        contexto_base = datetime.datetime.strptime(contexto_base_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return "ERROR: Formato de contexto base incorrecto (debe ser YYYY-MM-DD HH:MM)"

    peticion = peticion.lower()
    
    # --- Lógica de simulación para los casos más comunes ---
    
    # CASO 1: RESERVA RELATIVA (ej: "dentro de una hora")
    delta = datetime.timedelta()
    if "dentro de una hora" in peticion or "en una hora" in peticion:
        delta = datetime.timedelta(hours=1)
    elif "media hora" in peticion or "en 30 minutos" in peticion:
        delta = datetime.timedelta(minutes=30)
    
    # CASO 2: RESERVA PARA MAÑANA o ESTA NOCHE (Fácil)
    if "mañana a las" in peticion:
        # Mock: Asume 08:00 AM para la reserva de mañana
        hora_reserva = contexto_base + datetime.timedelta(days=1)
        return hora_reserva.strftime("%Y-%m-%d 08:00") 
    elif "esta noche" in peticion and contexto_base.hour < 19:
        # Mock: Asume 22:00 (10 PM) de hoy
        hora_reserva = contexto_base.replace(hour=22, minute=0)
        return hora_reserva.strftime("%Y-%m-%d %H:%M")
    elif "cinco y media" in peticion:
        # Mock: Asume las 17:30 (5:30 PM) de hoy o mañana
        target_time = contexto_base.replace(hour=17, minute=30, second=0, microsecond=0)
        if target_time < contexto_base:
             target_time += datetime.timedelta(days=1)
        return target_time.strftime("%Y-%m-%d %H:%M")

    # Si se encontró un delta (de tiempo relativo)
    if delta:
        hora_reserva = contexto_base + delta
        return hora_reserva.strftime("%Y-%m-%d %H:%M")
        
    # Resultado por defecto
    return contexto_base.strftime("%Y-%m-%d %H:%M") 
# ---------------------------------------------------
# --- Sección 3: El Endpoint de la API y Ejecución ---

@app.route('/predict_time', methods=['POST'])
def predict():
    """Endpoint para recibir la petición JSON y devolver la hora absoluta."""
    try:
        data = request.get_json()
        peticion = data.get('peticion')
        contexto_base = data.get('contexto_base')

        if not peticion or not contexto_base:
            return jsonify({"error": "Faltan los campos 'peticion' o 'contexto_base' en el JSON."}), 400

        # LLAMADA A LA FUNCIÓN DE SIMULACIÓN
        salida_absoluta = mock_predict_time(peticion, contexto_base)
        
        # Devolver el resultado en el formato API estándar
        return jsonify({
            "peticion_recibida": peticion,
            "contexto_base": contexto_base,
            "salida_absoluta": salida_absoluta,
            "simulacion": "ESTA RESPUESTA ES MOCK Y NO USA EL MODELO QWEN"
        })

    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    print("Servidor Mock listo para recibir peticiones POST en /predict_time.")
    # Usar host='0.0.0.0' para asegurar acceso si estás usando contenedores/VMs
    app.run(host='0.0.0.0', port=PORT, debug=False)
# ---------------------------------------------------