# Este código consiste en una estructura de dos procesos.
# En un núcleo se ejecuta el LiDAR y el IMU, y en otro núcleo se procesan los datos, con la
# posterior inferencia de la red neuronal. La comunicación entre procesos se realiza mediante 
# una cola de multiprocessing.

from servidor_web import lanzar_servidor_web # librería que contiene la función para lanzar el servidor web en un hilo separado

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
MAX_RANGE_MM = 3500.0 # Ajustado a 3500 según tu dataset de entrenamiento
MAX_GAP_DEG = 5

# Configuración para el cálculo de escalones
SECTOR_MIN_DEG  = 0        # zona angular donde aparece la escalera frontal
SECTOR_MAX_DEG  = 90
ANGULO_PISO_DEG = 90       # ángulo que apunta recto al piso (para medir H)

# Ángulos objetivo fijos: 0.0, 0.8, 1.6, ... siempre los mismos.
TARGET_ANGLES = np.arange(NUM_POINTS) * ANGLE_STEP

# ==========================================
# FUNCIONES UTILIZADAS
# ==========================================

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
 
    # --- NUEVA INTERPOLACIÓN ---
    # Limita las distancias para que ninguna sea mayor a MAX_RANGE_MM
    dist_interp[dist_interp > MAX_RANGE_MM] = MAX_RANGE_MM

    # Fuerza los ángulos entre 200° y 300° a tener el valor máximo
    mascara_ciega = (TARGET_ANGLES >= 200.0) & (TARGET_ANGLES <= 300.0)
    dist_interp[mascara_ciega] = MAX_RANGE_MM

    return dist_interp

def compensar_2d(vuelta_completa, imu_capturado):
    """
    Desfasa los ángulos crudos del LiDAR utilizando el vector de gravedad del IMU.
    Retorna la lista de puntos rotada y nivelada al horizonte real.
    """
    grav_x, grav_y, grav_z = imu_capturado['grav_promedio']
    
    angulo_gravedad_rad = np.arctan2(grav_y, grav_x)
    angulo_gravedad_deg = np.degrees(angulo_gravedad_rad)
    
    desfase = angulo_gravedad_deg - 90.0
    
    vuelta_corregida = []
    for angulo, distancia, calidad in vuelta_completa:
        angulo_nuevo = (angulo + desfase) % 360.0
        vuelta_corregida.append((angulo_nuevo, distancia, calidad))
        
    return vuelta_corregida

def calcular_roll_pitch(grav_x, grav_y, grav_z):
    """
    Calcula roll y pitch (en grados) a partir del vector de gravedad promediado.
    """
    roll_rad  = np.arctan2(grav_y, grav_z)
    pitch_rad = np.arctan2(-grav_x, np.sqrt(grav_y**2 + grav_z**2))
    return np.degrees(roll_rad), np.degrees(pitch_rad)

def load_model(device, pesos_path='lidar_cnn_model_B01.pth'):
    model = LidarCNN().to(device)
    if os.path.exists(pesos_path):
        model.load_state_dict(torch.load(pesos_path, map_location=device))
        print(f"Modelo cargado desde: {pesos_path}")
    else:
        print(f"No se encontró '{pesos_path}'. Se usará un modelo con pesos aleatorios.")
    model.eval()
    return model

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

# --- FUNCIONES DE CÁLCULO DE ESCALONES ---
def convertir_a_xz(angulos_deg, distancias, H, sector_min, sector_max, dist_invalida):
    """Convierte (ángulo, distancia) del sector de interés a coordenadas (x, z)."""
    mask = (
        (angulos_deg >= sector_min)
        & (angulos_deg <= sector_max)
        & (distancias < dist_invalida)
    )
    theta = angulos_deg[mask]
    r = distancias[mask]

    phi_rad = np.deg2rad(theta)
    x = r * np.cos(phi_rad)
    z = H - r * np.sin(phi_rad)

    return x, z

def calcular_angulo_superficie(x, z):
    """
    Calcula el ángulo de inclinación de una superficie ajustando una línea 
    a los puntos (x, z) mediante regresión lineal.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Requerimos un mínimo de puntos para que el ajuste sea válido y no haya ruido
    if len(x) < 5:
        return None
        
    # Ajuste polinomial de grado 1 (línea recta: z = mx + c)
    m, c = np.polyfit(x, z, 1)
    
    # La pendiente m es la tangente del ángulo, aplicamos arcotangente
    angulo_deg = np.degrees(np.arctan(m))
    
    return angulo_deg

def calcular_alturas_escalones(x, z, ventana=2, umbral_rango=15, min_puntos_plataforma=2, Trasero=False):
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    if len(x) == 0:
        return [], [], None

    orden = np.argsort(x)
    x_ord, z_ord = x[orden], z[orden]
    n = len(z_ord)

    medio = ventana // 2
    es_plano = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - medio)
        hi = min(n, i + medio + 1)
        rango_local = z_ord[lo:hi].max() - z_ord[lo:hi].min()
        es_plano[i] = rango_local < umbral_rango

    plataformas = []
    grupo_x, grupo_z = [], []

    for i in range(n):
        if es_plano[i]:
            grupo_x.append(x_ord[i])
            grupo_z.append(z_ord[i])
        else:
            if len(grupo_z) >= min_puntos_plataforma:
                plataformas.append((np.mean(grupo_x), np.median(grupo_z), len(grupo_z)))
            grupo_x, grupo_z = [], []

    if len(grupo_z) >= min_puntos_plataforma:
        plataformas.append((np.mean(grupo_x), np.median(grupo_z), len(grupo_z)))

    alturas = [abs(plataformas[i + 1][1] - plataformas[i][1]) for i in range(len(plataformas) - 1)]

    if len(alturas) >= 2:
        if Trasero:
            altura_escalon_estimada = np.mean(alturas[-2:])  
        else:
            altura_escalon_estimada = np.mean(alturas[:2])
    elif len(alturas) == 1:
        if Trasero:
            altura_escalon_estimada = alturas[-1]
        else:
            altura_escalon_estimada = alturas[0]
    else:
        altura_escalon_estimada = None

    return plataformas, alturas, altura_escalon_estimada

def detectar_obstaculo(x, z, umbral_z=150.0):
    """
    Filtra los puntos con z > umbral_z (en mm) y devuelve 
    la distancia x más cercana en la que se encuentra un obstáculo.
    """
    # Filtrar puntos que superen los 150 mm de altura
    mask_obstaculo = z > umbral_z
    x_obs = x[mask_obstaculo]
    z_obs = z[mask_obstaculo]
    
    # Si no hay puntos por encima de esa altura, no hay obstáculo
    if len(x_obs) == 0:
        return None
        
    # Encontramos el índice de la distancia X más cercana (menor valor absoluto)
    idx_min = np.argmin(np.abs(x_obs))
    
    # Retornamos la distancia x (conservando su signo)
    return x_obs[idx_min]

# ==========================================
# BUFFERS PARA LA MEDIA MÓVIL
# ==========================================
historial_grav_x = deque(maxlen=10)
historial_grav_y = deque(maxlen=10)
historial_grav_z = deque(maxlen=10)

# ==========================================
# VARIABLE COMPARTIDA (Memoria del Productor)
# ==========================================
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
        i2c = busio.I2C(board.SCL, board.SDA)
        bno = BNO08X_I2C(i2c)
        
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        bno.enable_feature(BNO_REPORT_GYROSCOPE)
        bno.enable_feature(BNO_REPORT_GRAVITY)

        for _ in range(10):
            historial_grav_x.append(0.0)
            historial_grav_y.append(-9.8)
            historial_grav_z.append(0.0)
        
        print("[IMU] BNO085 inicializado. Leyendo Accel, Gyro y Gravedad a 100Hz.")
        
    except Exception as e:
        print(f"[ERROR IMU] No se pudo inicializar el sensor. Revisa I2C. Detalle: {e}")
        return

    while True:
        try:
            accel_x, accel_y, accel_z = bno.acceleration
            gyro_x, gyro_y, gyro_z = bno.gyro
            grav_x, grav_y, grav_z = bno.gravity

            historial_grav_x.append(grav_x)
            historial_grav_y.append(grav_y)
            historial_grav_z.append(grav_z)

            prom_x = sum(historial_grav_x) / 10.0
            prom_y = sum(historial_grav_y) / 10.0
            prom_z = sum(historial_grav_z) / 10.0
            
            datos_imu_actuales = {
                'accel': (accel_x, accel_y, accel_z),
                'gyro':  (gyro_x, gyro_y, gyro_z),
                'grav_promedio':  (prom_x, prom_y, prom_z)
            }
            
            time.sleep(0.01)
            
        except OSError:
            pass

# ==========================================
# PROCESO 1: SUPER PRODUCTOR (Anclado al Núcleo 1)
# ==========================================
def productor_sensores(cola_datos):
    os.sched_setaffinity(0, {1})

    hilo_imu = threading.Thread(target=lector_imu_hilo, daemon=True)
    hilo_imu.start()

    lidar = PyRPlidar()
    lidar.connect(port="/dev/ttyUSB0", baudrate=460800, timeout=3)

    lidar.stop()
    time.sleep(0.2)
    lidar.lidar_serial._serial.reset_input_buffer()
    lidar.lidar_serial._serial.reset_output_buffer()
    time.sleep(0.3)

    lidar.set_motor_pwm(500)
    time.sleep(2) 

    scan_generator_func = lidar.start_scan()
    generador = scan_generator_func()

    proximo_inicio = None
    for scan in generador:
        if scan.start_flag:
            proximo_inicio = scan
            break

    print("[SISTEMA] Iniciando recolección sincronizada LiDAR + IMU (Inline Stream)...")

    while True:
        try:
            inicio_bucle = time.time()
            vuelta_completa = [(proximo_inicio.angle, proximo_inicio.distance, proximo_inicio.quality)]
            
            for scan in generador:
                if scan.start_flag:
                    proximo_inicio = scan  
                    break
                vuelta_completa.append((scan.angle, scan.distance, scan.quality))
            
            vuelta_completa.sort(key=lambda p: p[0])
            imu_capturado = datos_imu_actuales.copy() 
            
            paquete = (vuelta_completa, imu_capturado)
            cola_datos.put_nowait(paquete)
            
        except multiprocessing.queues.Full:
            pass
        except KeyboardInterrupt:
            break

    lidar.stop()
    lidar.set_motor_pwm(0)
    lidar.disconnect()

# ==========================================
# PROCESO 2: EL CEREBRO (Anclado al Núcleo 2)
# ==========================================
def consumidor_datos(cola_datos, cola_web):
    os.sched_setaffinity(0, {2})
    
    modelo = load_model(device='cpu', pesos_path='lidar_cnn_model_B01.pth')
    contador = 0

    # === MEMORIAS A CORTO PLAZO (Suavizado) ===
    buffer_altura_front = deque(maxlen=4)
    buffer_altura_back = deque(maxlen=4)
    
    buffer_angulo_front = deque(maxlen=5)
    buffer_angulo_back = deque(maxlen=5)

    buffer_obs_front = deque(maxlen=4)
    # ==========================================

    while True:
        paquete = cola_datos.get()
        inicio_bucle = time.time()
        vuelta_completa, imu_capturado = paquete 
        
        vuelta_nivelada = compensar_2d(vuelta_completa, imu_capturado)
        
        vuelta_interpolada_cruda    = interpolar_vuelta(vuelta_completa)
        vuelta_interpolada_nivelada = interpolar_vuelta(vuelta_nivelada)

        accel_x, accel_y, accel_z = imu_capturado['accel']
        grav_x, grav_y, grav_z = imu_capturado['grav_promedio']
        roll_deg, pitch_deg = calcular_roll_pitch(grav_x, grav_y, grav_z)

        X_lidar = torch.from_numpy(vuelta_interpolada_nivelada.astype('float32')).unsqueeze(0)  
        X_lidar_norm = normalizar_lidar(X_lidar)

        with torch.no_grad():
            salida = modelo(X_lidar_norm).squeeze(0) 
            
        prob_binarias = torch.sigmoid(salida).cpu().numpy()
        pred_binarias = (prob_binarias > 0.5).astype(int)

        # ==========================================
        # ANÁLISIS GEOMÉTRICO (POST-INFERENCIA CON PROMEDIOS)
        # ==========================================
        altura_est_front = None
        altura_est_back = None
        angulo_est_front = None
        angulo_est_back = None
        dist_obs_front = None

        # Verificar si alguna de las 4 primeras etiquetas es positiva para calcular H
        necesita_geometria = any(pred_binarias[:4] == 1)

        if necesita_geometria:
            idx_piso = np.argmin(np.abs(TARGET_ANGLES - ANGULO_PISO_DEG))
            H = vuelta_interpolada_nivelada[idx_piso]

        # --- 5. OBSTÁCULO FRONTAL Y TRASERO (Distancia X) ---
        if pred_binarias[4] == 1: 
            # Verificamos obstáculo hacia ADELANTE
            x_front, z_front = convertir_a_xz(TARGET_ANGLES, vuelta_interpolada_nivelada, H, 0, 90, MAX_RANGE_MM)
            dist_calculada_f = detectar_obstaculo(x_front, z_front, umbral_z=150.0)
            
            if dist_calculada_f is not None:
                buffer_obs_front.append(dist_calculada_f)
                
            if buffer_obs_front:
                dist_obs_front = sum(buffer_obs_front) / len(buffer_obs_front)
            
        else:
            buffer_obs_front.clear()

        # --- 1. SUPERFICIE FRONTAL (Ángulo) ---
        if pred_binarias[0] == 1:
            # CAMBIO: Usar 55 y 65 en lugar de SECTOR_MIN_DEG y SECTOR_MAX_DEG
            x_front, z_front = convertir_a_xz(TARGET_ANGLES, vuelta_interpolada_nivelada, H, 45, 70, MAX_RANGE_MM)
            ang_calculado = calcular_angulo_superficie(x_front, z_front)
            
            if ang_calculado is not None:
                buffer_angulo_front.append(ang_calculado)
                
            if buffer_angulo_front:
                angulo_est_front = sum(buffer_angulo_front) / len(buffer_angulo_front)
        else:
            buffer_angulo_front.clear()

        # --- 2. SUPERFICIE TRASERA (Ángulo) ---
        if pred_binarias[1] == 1:
            # CAMBIO: Usar 115 y 125 en lugar de sumar 90 a SECTOR_MIN/MAX
            x_back, z_back = convertir_a_xz(TARGET_ANGLES, vuelta_interpolada_nivelada, H, 105, 120, MAX_RANGE_MM)
            ang_calculado = calcular_angulo_superficie(x_back, z_back)
            
            if ang_calculado is not None:
                buffer_angulo_back.append(-ang_calculado)
                
            if buffer_angulo_back:
                angulo_est_back = sum(buffer_angulo_back) / len(buffer_angulo_back)
                angulo_est_back += 1
        else:
            buffer_angulo_back.clear()

        # --- 3. ESCALERA FRONTAL (Altura) ---
        if pred_binarias[2] == 1: 
            x_front, z_front = convertir_a_xz(TARGET_ANGLES, vuelta_interpolada_nivelada, H, SECTOR_MIN_DEG, SECTOR_MAX_DEG, MAX_RANGE_MM)
            _, _, altura_calculada = calcular_alturas_escalones(x_front, z_front)
            
            if altura_calculada is not None:
                buffer_altura_front.append(altura_calculada)
            
            if buffer_altura_front:
                altura_est_front = sum(buffer_altura_front) / len(buffer_altura_front)
        else:
            buffer_altura_front.clear()

        # --- 4. ESCALERA TRASERA (Altura) ---
        if pred_binarias[3] == 1: 
            x_back, z_back = convertir_a_xz(TARGET_ANGLES, vuelta_interpolada_nivelada, H, 90 + SECTOR_MIN_DEG, 90 + SECTOR_MAX_DEG, MAX_RANGE_MM)
            _, _, altura_calculada = calcular_alturas_escalones(x_back, z_back, Trasero=True)
            
            if altura_calculada is not None:
                buffer_altura_back.append(altura_calculada)
            
            if buffer_altura_back:
                altura_est_back = sum(buffer_altura_back) / len(buffer_altura_back)
        else:
            buffer_altura_back.clear()
        # ==========================================

        final_bucle = time.time()

        nombres_etiquetas = [
            "sup_frontal", "sup_trasera", "esc_frontal",
            "esc_trasera", "obstaculo"
        ]

        # Impresiones en consola controladas
        if contador == 15:
            print(f"\n=== INFERENCIA EN TIEMPO REAL ===")
            print(f"Tiempo Procesamiento: {final_bucle - inicio_bucle:.4f} segundos")
            for i, nombre in enumerate(nombres_etiquetas):
                print(f"{nombre.ljust(15)}: pred={pred_binarias[i]} (prob={prob_binarias[i]:.4f})")
                
            print("-" * 33)
            if angulo_est_front is not None:
                print(f">> Ángulo Suelo Frontal: {angulo_est_front:+.1f}°")
            if angulo_est_back is not None:
                print(f">> Ángulo Suelo Trasero:  {angulo_est_back:+.1f}°")
            if altura_est_front is not None:
                print(f">> Altura Escalón Frontal:{altura_est_front:.1f} mm")
            if altura_est_back is not None:
                print(f">> Altura Escalón Trasero: {altura_est_back:.1f} mm")
            if dist_obs_front is not None:
                print(f">> Distancia Obstáculo Frontal: {dist_obs_front:.1f} mm")
                
            print("=================================\n")
            contador = 0
        
        contador += 1

        try:
            cola_web.put_nowait({
                "cruda": vuelta_completa,
                "nivelada": vuelta_nivelada,
                "interpolada_cruda": vuelta_interpolada_cruda.tolist(),
                "interpolada": vuelta_interpolada_nivelada.tolist(),
                "imu": imu_capturado,
                "imu_roll_deg": roll_deg,
                "imu_pitch_deg": pitch_deg,
                "altura_esc_front": altura_est_front, 
                "altura_esc_back": altura_est_back,
                "angulo_sup_front": angulo_est_front,
                "angulo_sup_back": angulo_est_back
            })
        except multiprocessing.queues.Full: pass

# ==========================================
# ARRANQUE PRINCIPAL DEL PROGRAMA
# ==========================================
if __name__ == '__main__':
    cola_compartida = multiprocessing.Queue(maxsize=3) 
    cola_web = multiprocessing.Queue(maxsize=1) 

    proceso_web = multiprocessing.Process(target=lanzar_servidor_web, args=(cola_web,))
    proceso_sensores = multiprocessing.Process(target=productor_sensores, args=(cola_compartida,))
    
    proceso_cerebro = multiprocessing.Process(target=consumidor_datos, args=(cola_compartida, cola_web))

    try:
        proceso_web.start()
        proceso_cerebro.start()
        proceso_sensores.start()
        
        proceso_sensores.join()
        proceso_cerebro.join()
        proceso_web.join()
        
    except KeyboardInterrupt:
        print("\n[SISTEMA] Deteniendo procesos de forma segura...")
        proceso_sensores.terminate()
        proceso_cerebro.terminate()
        proceso_web.terminate()