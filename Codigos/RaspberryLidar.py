import numpy as np
import matplotlib.pyplot as plt
import json
import time
import paho.mqtt.client as mqtt
import random


# CALLBACKS                                                                                         # Para manejar estado del codigo ejecutado en raspi

def on_connect(client, userdata, flags, rc):                                                        # Cuando la raspi se conecta al servidor
    if rc == 0:
        print("✅ Conectado exitosamente al Broker MQTT")
    else:
        print(f"❌ Error de conexión. Código de retorno: {rc}")

def on_disconnect(client, userdata, rc):                                                            # Por si la raspi se desconecta del servidor
    if rc != 0:
        print("❌ Desconexión inesperada del Broker.")
    else:
        print("ℹ✅ Desconectado del Broker manualmente.")

def on_publish(client, userdata, mid):                                                              # Para cada mensaje enviado (deberian ser 5 por segundo)
    print(f"--> Mensaje enviado con ID: {mid}")


# CONFIG

BROKER = "localhost"                                                                                # o IP si quieres externo
TOPIC = "lidar/data"
H = 0.5                                                                                             # Altura donde se ubica Lidar en metros, altura de la rodilla unos 50 cm
MIN_ANGULO = -90
MAX_ANGULO = 90
RESOLUCION = MAX_ANGULO - MIN_ANGULO


# Funciones

def conectar_mqtt():                                                                                # Conecta MQTT, agrega los callback y maneja errores
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    
    print(f"Intentando conectar a broker: {BROKER}...")
    try:
        client.connect(BROKER, 1883, 60)
    except Exception as e:
        print(f"No se pudo iniciar la conexión: {e}")

    client.loop_start()
    client.connect(BROKER, 1883, 60)
    return client

def publicar_lidar(client, clase, theta, r):                                                               # Para enviar un mensaje, con manejo de errores
    data = {
        "clase": clase,
        "theta": theta.tolist(),
        "r": r.tolist()
    }

    payload = json.dumps(data)
    result = client.publish(TOPIC, payload)
    status = result[0]
    if status != 0:
        print(f"Fallo al enviar mensaje al topic {TOPIC}")

def generar_suelo():
    theta = np.linspace(MIN_ANGULO, MAX_ANGULO, RESOLUCION)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = -H / np.sin(np.radians(theta))
    mask = (r > 0) & (r <= 12) & np.isfinite(r)
    return theta[mask], r[mask]

def generar_rampa(x_0=5, m=0.5):
    theta_full = np.linspace(MIN_ANGULO, MAX_ANGULO, RESOLUCION)                                                          # Suelo antes de la rampa
    theta_rad = np.radians(theta_full)
    
    r_suelo = -H / np.sin(np.radians(theta_full))
    x_suelo = r_suelo * np.cos(theta_rad)
    mask_suelo = (x_suelo >= 0) & (x_suelo < x_0) & (r_suelo > 0) & (r_suelo < 20)
    
    r_rampa = (-H - m*x_0) / (np.sin(theta_rad) - m*np.cos(theta_rad))
    x_rampa = r_rampa * np.cos(theta_rad)                                                           # Filtrar para quedarnos solo con la parte de la rampa x >= x_0
    mask_rampa = np.isfinite(r_rampa) & (r_rampa >= 0) & (x_rampa >= x_0)

    theta = np.concatenate((theta_full[mask_suelo], theta_full[mask_rampa]))
    r = np.concatenate((r_suelo[mask_suelo], r_rampa[mask_rampa]))
    
    dist_mask = (r > 0) & (r <= 12) & np.isfinite(r)

    return theta[dist_mask], r[dist_mask]

def generar_escalera(x_0=2, largo_escalon=0.4, alto_escalon=0.2, escalones=10):

    theta = np.linspace(MIN_ANGULO, MAX_ANGULO, RESOLUCION)                                                               # Angulos del LiDAR (180° completo)

    segmentos = []                                                                                  # Construir segmentos de la escena
    segmentos.append(((0, -H), (x_0, -H)))                                                          # Suelo antes de la escalera

    for i in range(escalones):
        x_ini = x_0 + i * largo_escalon
        x_fin = x_0 + (i + 1) * largo_escalon

        y_base = -H + i * alto_escalon
        y_top = -H + (i + 1) * alto_escalon

        segmentos.append(((x_ini, y_base), (x_ini, y_top)))                                         # Cara vertical
        segmentos.append(((x_ini, y_top), (x_fin, y_top)))                                          # Parte horizontal

    r = []

    for th in theta:
        th_rad = np.radians(th)
        dx = np.cos(th_rad)
        dy = np.sin(th_rad)

        distancias = []

        for (x1, y1), (x2, y2) in segmentos:
            A = np.array([[dx, -(x2 - x1)],                                                         # Resolver interseccion rayo-segmento
                          [dy, -(y2 - y1)]])
            b = np.array([x1, y1])

            try:
                sol = np.linalg.solve(A, b)
                r_i, u = sol
            except:
                continue

            if r_i >= 0 and 0 <= u <= 1:                                                            # Condiciones de interseccion valida
                distancias.append(r_i)

        if len(distancias) > 0:
            r.append(min(distancias))
        else:
            r.append(np.nan)

    r_array = np.array(r)
    mask = np.isfinite(r_array) & (r_array > 0)
    
    return theta[mask], r_array[mask]

def graficar(theta, r, titulo):                                                                     # Solamente para debuguear
    theta_rad = np.radians(theta)

    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)

    mask = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(r) &
        (r >= 0) &
        (x >= 0) &
        (np.abs(x) < 20) &
        (np.abs(y) < 20)
    )

    x = x[mask]
    y = y[mask]

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y + 0.5, s=30, label='Mediciones LiDAR')
    plt.scatter(0, 0.5, c='red', label='LiDAR')

    plt.title(titulo)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    
    client = conectar_mqtt()

    theta_suelo, r_suelo = generar_suelo()
    theta_rampa, r_rampa = generar_rampa()
    theta_escalera, r_escalera = generar_escalera()
    
    # graficar(theta_suelo, r_suelo, "grafico")
    # graficar(theta_rampa, r_rampa, "grafico")
    # graficar(theta_escalera, r_escalera, "grafico")

    while True:
        
        n = random.randint(1, 3)
        if n == 1:
            publicar_lidar(client, "Suelo", theta_suelo, r_suelo)
        elif n == 2:
            publicar_lidar(client, "Rampa", theta_rampa, r_rampa)
        else:
            publicar_lidar(client, "Escalera", theta_escalera, r_escalera)
        
        time.sleep(0.2)