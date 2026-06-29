# Esto código consiste en una estructura de dos procesos
# En un nucleo se ejecuta el LiDAR y el IMU, y en otro núcleo se procesan los datos, con la
# posterior inferencia de la red neuronal. La comunicación entre procesos se realiza mediante 
# una cola de multiprocessing.


from servidor_web import lanzar_servidor_web # libreria que contiene la función para lanzar el servidor web en un hilo separado

import multiprocessing
import threading
import time
import os
import board
import busio
from pyrplidar import PyRPlidar
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GRAVITY
)
from adafruit_bno08x.i2c import BNO08X_I2C
from collections import deque
import numpy as np
from LidarRed import LidarCNN 
import torch
import torch.nn as nn


# ==========================================
# CONFIGURACIONES
# ==========================================

NUM_POINTS = 450
ANGLE_STEP = 0.8
MAX_RANGE_MM = 12000
MAX_GAP_DEG = 5

# Ángulos objetivo fijos: 0.0, 0.8, 1.6, ... siempre los mismos.
TARGET_ANGLES = np.arange(NUM_POINTS) * ANGLE_STEP

# ==========================================
# FUNCIONES UTILIZADAS
# ==========================================

# Funcion de interpolación de vuelta del LiDAR
# Recibe una vuelta completa de datos (lista de tuplas) y devuelve un vector fijo de 450 distancias.
def interpolar_vuelta(vuelta):
    """Vector fijo de 450 distancias. Interpola con puntos válidos; huecos
    grandes (> MAX_GAP_DEG) se asignan a MAX_RANGE_MM (sin retorno / lejano)."""
    angles = np.array([p[0] for p in vuelta], dtype=float)
    distances = np.array([p[1] for p in vuelta], dtype=float)
 
    valido = distances > 0.0
    ang_v = angles[valido]
    dist_v = distances[valido]
 
    if ang_v.size < 2:
        return np.full(NUM_POINTS, MAX_RANGE_MM)
 
    orden = np.argsort(ang_v)
    ang_v = ang_v[orden]
    dist_v = dist_v[orden]
 
    dist_interp = np.interp(TARGET_ANGLES, ang_v, dist_v, period=360)
 
    # Huecos grandes en la cobertura válida -> rango máximo.
    ext = np.concatenate([ang_v, [ang_v[0] + 360.0]])
    for i in range(ang_v.size):
        lo, hi = ext[i], ext[i + 1]
        if hi - lo > MAX_GAP_DEG:
            dentro = ((TARGET_ANGLES > lo) & (TARGET_ANGLES < hi)) | (
                (TARGET_ANGLES + 360.0 > lo) & (TARGET_ANGLES + 360.0 < hi)
            )
            dist_interp[dentro] = MAX_RANGE_MM
 
    return dist_interp

def compensar_2d(vuelta_completa, imu_capturado):
    """
    Desfasa los ángulos crudos del LiDAR utilizando el vector de gravedad del IMU.
    Retorna la lista de puntos rotada y nivelada al horizonte real.
    """
    # 1. Extraemos la gravedad promediada
    grav_x, grav_y, grav_z = imu_capturado['grav_promedio']
    
    # 2. Calculamos el ángulo actual de la gravedad (en grados)
    # Estando nivelado: G_x = 0, G_y = -9.8 --> angulo = -90 grados
    angulo_gravedad_rad = np.arctan2(grav_y, grav_x)
    angulo_gravedad_deg = np.degrees(angulo_gravedad_rad)
    
    # 3. Calculamos la desviación respecto al ideal (-90)
    desfase = angulo_gravedad_deg - 90.0
    
    # 4. Aplicamos el desfase angular a cada tupla
    vuelta_corregida = []
    for angulo, distancia, calidad in vuelta_completa:
        angulo_nuevo = (angulo + desfase) % 360.0
        vuelta_corregida.append((angulo_nuevo, distancia, calidad))
        
    return vuelta_corregida


def calcular_roll_pitch(grav_x, grav_y, grav_z):
    """
    Calcula roll y pitch (en grados) a partir del vector de gravedad promediado.
    Convención estándar (aeroespacial), asumiendo el eje Z del sensor "hacia arriba":
        roll  = atan2(grav_y, grav_z)
        pitch = atan2(-grav_x, sqrt(grav_y^2 + grav_z^2))
    NOTA: si la orientación física de montaje del BNO085 es distinta, los signos o
    los ejes usados aquí pueden necesitar ajuste.
    """
    roll_rad  = np.arctan2(grav_y, grav_z)
    pitch_rad = np.arctan2(-grav_x, np.sqrt(grav_y**2 + grav_z**2))
    return np.degrees(roll_rad), np.degrees(pitch_rad)


def load_model(device, pesos_path='lidar_cnn_model.pth'):
    model = LidarCNN().to(device)
    if os.path.exists(pesos_path):
        model.load_state_dict(torch.load(pesos_path, map_location=device))
        print(f"Modelo cargado desde: {pesos_path}")
    else:
        print(f"No se encontró '{pesos_path}'. Se usará un modelo con pesos aleatorios.")
    model.eval()
    return model


def normalizar_datos(X_lidar, X_imu):
    X_lidar_min = X_lidar.min(dim=1, keepdim=True)[0]
    X_lidar_max = X_lidar.max(dim=1, keepdim=True)[0]
    X_lidar_norm = (X_lidar - X_lidar_min) / (X_lidar_max - X_lidar_min + 1e-8)

    X_imu_mean = X_imu.mean(dim=1, keepdim=True)
    X_imu_std = X_imu.std(dim=1, keepdim=True)
    X_imu_norm = (X_imu - X_imu_mean) / (X_imu_std + 1e-8)

    return X_lidar_norm, X_imu_norm

def normalizar_lidar(X_lidar):
    """
    X_lidar: tensor de forma (B, N) o (N,).
    Normaliza cada fila usando su propio min/max (independiente entre filas).
    """
    X_lidar = X_lidar.float()
    if X_lidar.dim() == 1:
        X_lidar = X_lidar.unsqueeze(0)   # (N,) -> (1, N)

    X_lidar_min = X_lidar.min(dim=1, keepdim=True)[0]
    X_lidar_max = X_lidar.max(dim=1, keepdim=True)[0]
    X_lidar_norm = (X_lidar - X_lidar_min) / (X_lidar_max - X_lidar_min + 1e-8)

    return X_lidar_norm.float()

def normalizar_datos(X_imu):
    X_imu = X_imu.float()
    if X_imu.dim() == 1:
        X_imu = X_imu.unsqueeze(0)
    X_imu_mean = X_imu.mean(dim=1, keepdim=True)[0]
    X_imu_std = X_imu.std(dim=1, keepdim=True)[0]
    X_imu_norm = (X_imu - X_imu_mean) / (X_imu_std + 1e-8)
    return X_imu_norm.float()


# ==========================================
# BUFFERS PARA LA MEDIA MÓVIL
# ==========================================
# Guardarán exactamente las últimas 10 muestras
historial_grav_x = deque(maxlen=10)
historial_grav_y = deque(maxlen=10)
historial_grav_z = deque(maxlen=10)

# ==========================================
# VARIABLE COMPARTIDA (Memoria del Productor)
# ==========================================
# Inicializamos con valores neutros (gravedad apuntando hacia abajo en Y por defecto)
# (SE PUEDEN CAMBIAR EL MODO EN EL QUE SE ALMCENA LOS DATOS SI SE DESEA)
datos_imu_actuales = {
    'accel': (0.0, 0.0, 0.0),
    'gyro':  (0.0, 0.0, 0.0),
    'grav_promedio':  (0.0, -9.8, 0.0)
}

# ==========================================
# HILO SECUNDARIO (Lee el IMU a 100Hz)
# ==========================================
def lector_imu_hilo():
    global datos_imu_actuales
    
    try:
        # 1. Inicialización del bus I2C y el sensor
        i2c = busio.I2C(board.SCL, board.SDA)
        bno = BNO08X_I2C(i2c)
        
        # 2. Habilitación de los 3 reportes solicitados
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        bno.enable_feature(BNO_REPORT_GYROSCOPE)
        bno.enable_feature(BNO_REPORT_GRAVITY)

        # Pre-llenamos el buffer para que los primeros 100ms no promedien con ceros
        for _ in range(10):
            historial_grav_x.append(0.0)
            historial_grav_y.append(-9.8)
            historial_grav_z.append(0.0)
        
        print("[IMU] BNO085 inicializado. Leyendo Accel, Gyro y Gravedad a 100Hz.")
        
    except Exception as e:
        print(f"[ERROR IMU] No se pudo inicializar el sensor. Revisa I2C. Detalle: {e}")
        return # Termina el hilo de forma segura si no hay hardware

    # 3. Bucle infinito de lectura
    while True:
        try:
            # Extraemos las tuplas (x, y, z) directamente del hardware
            accel_x, accel_y, accel_z = bno.acceleration
            gyro_x, gyro_y, gyro_z = bno.gyro
            grav_x, grav_y, grav_z = bno.gravity

            # Agregamos la nueva lectura al historial (empuja y borra la más vieja)
            historial_grav_x.append(grav_x)
            historial_grav_y.append(grav_y)
            historial_grav_z.append(grav_z)

            # Calculamos el promedio de los últimos 10 datos (aplasta el ruido)
            prom_x = sum(historial_grav_x) / 10.0
            prom_y = sum(historial_grav_y) / 10.0
            prom_z = sum(historial_grav_z) / 10.0
            
            # Actualizamos el DICCIONARIO GLOBAL (SE PUEDEN CAMBIAR EL MODO EN EL QUE SE ALMCENA LOS DATOS SI SE DESEA)
            datos_imu_actuales = {
                'accel': (accel_x, accel_y, accel_z),
                'gyro':  (gyro_x, gyro_y, gyro_z),
                'grav_promedio':  (prom_x, prom_y, prom_z)
            }
            
            # Retardo exacto para mantener la frecuencia de 100Hz (10ms)
            time.sleep(0.01)
            
        except OSError:
            # Los buses I2C a veces tienen micro-desconexiones físicas.
            # Capturamos el error silenciosamente para que el hilo no muera
            # y reintente en el siguiente ciclo.
            pass

# ==========================================
# PROCESO 1: SUPER PRODUCTOR (Anclado al Núcleo 1)
# ==========================================
def productor_sensores(cola_datos):
    os.sched_setaffinity(0, {1})

    # Arrancamos el Hilo del IMU en el fondo
    hilo_imu = threading.Thread(target=lector_imu_hilo, daemon=True)
    hilo_imu.start()

    # Inicializamos el LiDAR
    lidar = PyRPlidar()
    lidar.connect(port="/dev/ttyUSB0", baudrate=460800, timeout=3)

    # LIMPIEZA DE BUFFER (Crucial para reiniciar sin errores)
    lidar.stop()
    time.sleep(0.2)
    lidar.lidar_serial._serial.reset_input_buffer()
    lidar.lidar_serial._serial.reset_output_buffer()
    time.sleep(0.3)

    lidar.set_motor_pwm(500)
    time.sleep(2) # Esperamos a que el motor alcance velocidad estable

    scan_generator_func = lidar.start_scan()
    
    # ¡OPTIMIZACIÓN SUPREMA!: Instanciamos el iterador de flujo contínuo una sola vez afuera
    generador = scan_generator_func()

    # Enganchamos el flujo buscando el PRIMER inicio oficial antes de entrar al while
    proximo_inicio = None
    for scan in generador:
        if scan.start_flag:
            proximo_inicio = scan
            break

    print("[SISTEMA] Iniciando recolección sincronizada LiDAR + IMU (Inline Stream)...")

    # 3. Bucle principal unificado (Todo en una sola estructura lineal)
    while True:
        try:
            # Inicializamos la lista directamente con el punto que cerró la vuelta anterior
            # Así no perdemos muestras y mantenemos el desfase de hardware controlado
            vuelta_completa = [(proximo_inicio.angle, proximo_inicio.distance, proximo_inicio.quality)]
            
            # RECOLECCIÓN PURA EN UN SOLO BUCLE EN LÍNEA
            for scan in generador:
                if scan.start_flag:
                    # Capturamos el inicio de la SIGUIENTE vuelta y rompemos el bucle actual
                    proximo_inicio = scan  
                    break
                # Guardamos los puntos intermedios a velocidad nativa de C
                vuelta_completa.append((scan.angle, scan.distance, scan.quality))
            
            # Ordenamos los puntos por ángulo (vital para interpolar_vuelta en el Núcleo 2)
            vuelta_completa.sort(key=lambda p: p[0])
            
            # Capturamos la foto exacta del IMU promediado en este instante de cierre
            imu_capturado = datos_imu_actuales.copy() 
            
            # Enviamos el paquete combinado al Cerebro
            paquete = (vuelta_completa, imu_capturado)
            cola_datos.put_nowait(paquete)
            
        except multiprocessing.queues.Full:
            # Si el Cerebro se satura, saltamos el paquete para proteger los tiempos reales
            pass
        except KeyboardInterrupt:
            break

    # Apagado seguro
    lidar.stop()
    lidar.set_motor_pwm(0)
    lidar.disconnect()


# ==========================================
# PROCESO 2: EL CEREBRO (Anclado al Núcleo 2)
# ==========================================
def consumidor_datos(cola_datos, cola_web):
    os.sched_setaffinity(0, {2})
    
    # cargar modelo (antes del while)
    modelo = load_model(device='cpu', pesos_path='lidar_cnn_model.pth')
    contador = 0

    while True:
        # Esperamos y desempaquetamos los datos
        paquete = cola_datos.get()  
        vuelta_completa, imu_capturado = paquete 
        
        # NIVELACIÓN: Corregimos los ángulos con la gravedad del IMU
        vuelta_nivelada = compensar_2d(vuelta_completa, imu_capturado)
        
        # INTERPOLACIÓN: Forzamos los puntos al vector fijo de 450 elementos
        # - vuelta_interpolada_cruda:    LiDAR crudo (sin nivelar por IMU), interpolado
        # - vuelta_interpolada_nivelada: LiDAR nivelado por IMU, interpolado
        vuelta_interpolada_cruda    = interpolar_vuelta(vuelta_completa)
        vuelta_interpolada_nivelada = interpolar_vuelta(vuelta_nivelada)

        # ROLL / PITCH: derivados del vector de gravedad promediado
        accel_x, accel_y, accel_z = imu_capturado['accel']
        grav_x, grav_y, grav_z = imu_capturado['grav_promedio']
        roll_deg, pitch_deg = calcular_roll_pitch(grav_x, grav_y, grav_z)

        # La inferencia recibe un vector de 450 distancias crudos interpolados mas 8 datos del imu (sin giroscopio) y devuelve un diccionario con la predicción
        X_lidar = torch.from_numpy(vuelta_interpolada_nivelada.astype('float32')).unsqueeze(0)  # (1, 450)
        X_imu = torch.from_numpy(np.array([
            accel_x, accel_y, accel_z, grav_x, grav_y, grav_z, roll_deg, pitch_deg
        ], dtype=np.float32)).unsqueeze(0)  # (1, 8)

        # Normalizamos los datos de entrada
        X_lidar_norm = normalizar_lidar(X_lidar)
        X_imu_norm = normalizar_datos(X_imu)

        # Inferencia con el modelo
        with torch.no_grad():
            salida = modelo(X_lidar_norm).squeeze(0)
            prediccion = salida.cpu().numpy()  # Convertimos a numpy para enviar al web [9 ETIQUETAS {sup_frontal, sup_trasera, esc_frontal, esc_trasera, obstaculo, ang_sup_frontal, ang_sup_trasera, altura_escalones, dist_obstaculo}]

        prob_binarias = torch.sigmoid(salida[:5]).cpu().numpy()
        valores_regresion = torch.sigmoid(salida[5:]).cpu().numpy()
        pred_binarias = (prob_binarias > 0.5).astype(int)

        nombres_etiquetas = [
            "sup_frontal (bin)", "sup_trasera (bin)", "esc_frontal (bin)",
            "esc_trasera (bin)", "obstaculo (bin)", "ang_sup_frontal (float)",
            "ang_sup_trasera (float)", "altura_escalones (float)", "dist_obstaculo (float)"
        ]

        if contador == 15:
            print(f"\n=== INFERENCIA RANDOM EN VALIDACIÓN ===")
            for i, nombre in enumerate(nombres_etiquetas[:5]):
                print(f"{nombre}: pred={pred_binarias[i]} (prob={prob_binarias[i]:.4f})")
            for i, nombre in enumerate(nombres_etiquetas[5:], start=5):
                if nombre == "ang_sup_frontal (float)" or nombre == "ang_sup_trasera (float)":
                    print(f"{nombre}: pred={(valores_regresion[i-5]*30-15):.4f} [°]")
                if nombre == "dist_obstaculo (float)":
                    print(f"{nombre}: pred={((valores_regresion[i-5]*2.7+0.3))*100:.4f} [cm]")
                if nombre == "altura_escalones (float)":
                    print(f"{nombre}: pred={(valores_regresion[i-5]*30):.4f} [cm]")
            print("============================================\n")
            contador = 0
        
        contador += 1

        # 4. EXPORTACIÓN WEB (Todo el paquete en un diccionario de una línea)
        try:
            cola_web.put_nowait({
                "cruda": vuelta_completa,
                "nivelada": vuelta_nivelada,
                "interpolada_cruda": vuelta_interpolada_cruda.tolist(),
                "interpolada": vuelta_interpolada_nivelada.tolist(),
                "imu": imu_capturado,
                "imu_roll_deg": roll_deg,
                "imu_pitch_deg": pitch_deg
            })
        except multiprocessing.queues.Full: pass



# ==========================================
# ARRANQUE PRINCIPAL DEL PROGRAMA
# ==========================================
if __name__ == '__main__':
    # 1. Definición de las colas
    cola_compartida = multiprocessing.Queue(maxsize=3) 
    cola_web = multiprocessing.Queue(maxsize=1) 

    # 2. Creación de los procesos
    proceso_web = multiprocessing.Process(target=lanzar_servidor_web, args=(cola_web,))
    proceso_sensores = multiprocessing.Process(target=productor_sensores, args=(cola_compartida,))
    
    # Se inyectan AMBAS colas al cerebro
    proceso_cerebro = multiprocessing.Process(target=consumidor_datos, args=(cola_compartida, cola_web))

    try:
        # 3. Arranque de los procesos
        proceso_web.start()
        proceso_cerebro.start()
        proceso_sensores.start()
        
        # 4. Mantener vivo el script principal
        proceso_sensores.join()
        proceso_cerebro.join()
        proceso_web.join()
        
    except KeyboardInterrupt:
        print("\n[SISTEMA] Deteniendo procesos de forma segura...")
        proceso_sensores.terminate()
        proceso_cerebro.terminate()
        proceso_web.terminate()