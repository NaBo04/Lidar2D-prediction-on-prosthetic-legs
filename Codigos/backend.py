import time
import random
from supabase import create_client, Client
import json
import queue
import paho.mqtt.client as mqtt


# CONFIG

SUPABASE_URL = "https://ggrpykvwguknefqbhlvx.supabase.co"                                           # Credenciales
SUPABASE_KEY = "sb_publishable_AoTEVrhfuEzJVbHpgpbIQA_RQ2YdIqB"
NOMBRE_TABLA = "Datos y Predicciones"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BROKER = "100.71.137.62"
TOPIC = "lidar/data"
DATA_QUEUE = queue.Queue()


# CALLBACKS

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Backend conectado al Broker MQTT")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Error de conexión MQTT: {rc}")
        
def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"❌ Desconexión inesperada (Código: {rc}).")
    else:
        print("✅ Desconectado del broker de forma manual.")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if DATA_QUEUE.full():
            try:
                DATA_QUEUE.get_nowait()
            except queue.Empty:
                pass
        DATA_QUEUE.put(payload)
        print(f"--> Mensaje recibido")
    except Exception as e:
        print(f"⚠️ Error procesando mensaje: {e}")


# Funciones

def conectar_mqtt():
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.connect(BROKER, 1883, 60)
    client.loop_start()

    return client

def recibir_mensaje(cliente):
    
    if not DATA_QUEUE.empty():
        datos = None
        while not DATA_QUEUE.empty():
            datos = DATA_QUEUE.get()
        
        clase = datos.get("clase", "Sin Clase")
        theta = datos.get("theta", [])
        r = datos.get("r", [])
        
        return [clase, theta, r]

def actualizar_base_de_datos(clase, theta, r):    
    datos_insercion = {
        "clase": clase,
        "lista_theta": theta,
        "lista_r": r
    }
    
    try:
        supabase.table(NOMBRE_TABLA).insert(datos_insercion).execute()
        print(f"[{time.strftime('%H:%M:%S')}] Clase enviada: {clase}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    cliente = conectar_mqtt()
    cuenta = 0
    
    try:
        while True:
            datos_recibidos = recibir_mensaje(cliente)
            
            if datos_recibidos:
                clase, theta, r = datos_recibidos
                cuenta += 1
                if cuenta == 5:
                    cuenta = 0
                    actualizar_base_de_datos(clase, theta, r)
            
    except KeyboardInterrupt:
        print("\nTransmisión detenida.")
        cliente.loop_stop()
        cliente.disconnect()