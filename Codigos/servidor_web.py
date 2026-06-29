import os, json, csv, datetime
from flask import Flask, Response, stream_with_context, request, jsonify

app = Flask(__name__)
cola_global    = None
filas_guardadas = 0          # contador de frames guardados en esta sesión
CSV_CRUDO_PATH    = 'dataset_lidar_crudo.csv'     # LiDAR crudo interpolado + IMU en vivo
CSV_NIVELADO_PATH = 'dataset_lidar_nivelado.csv'  # LiDAR nivelado interpolado + etiquetas

# ==========================================
# RUTAS FLASK
# ==========================================

@app.route('/')
def index():
    return HTML_PAGE


@app.route('/stream')
def stream_datos():
    def generate():
        while True:
            try:
                paquete = cola_global.get(timeout=0.4)
                datos = {
                    'cruda':       [[float(p[0]), float(p[1]), int(p[2])]
                                    for p in paquete['cruda']],
                    'nivelada':    [[float(p[0]), float(p[1]), int(p[2])]
                                    for p in paquete['nivelada']],
                    'interpolada':       [float(v) for v in paquete['interpolada']],
                    'interpolada_cruda': [float(v) for v in paquete['interpolada_cruda']],
                    'imu': {
                        'accel': [float(v) for v in paquete['imu']['accel']],
                        'gyro':  [float(v) for v in paquete['imu']['gyro']],
                        'grav':  [float(v) for v in paquete['imu']['grav_promedio']],
                        'roll_deg':  float(paquete['imu_roll_deg']),
                        'pitch_deg': float(paquete['imu_pitch_deg'])
                    }
                }
                yield 'data: ' + json.dumps(datos) + '\n\n'
            except Exception:
                yield ': keepalive\n\n'

    resp = Response(stream_with_context(generate()), mimetype='text/event-stream')
    resp.headers['Cache-Control']     = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/guardar', methods=['POST'])
def guardar_frame():
    """
    Recibe { frame: {...}, labels: {...} } y añade una fila a CADA UNO de los
    dos datasets:
      - CSV_CRUDO_PATH:    LiDAR crudo interpolado (sin nivelar por IMU) + IMU en vivo + etiquetas.
      - CSV_NIVELADO_PATH: LiDAR nivelado interpolado + etiquetas (sin datos de IMU).
    Crea los CSV con encabezados si no existían.
    Devuelve { status, total_filas } para que el frontend actualice el contador.
    """
    global filas_guardadas
    data  = request.get_json(force=True)
    frame = data.get('frame', {})
    etiq  = data.get('labels', {})

    imu   = frame.get('imu', {})
    accel = imu.get('accel', [0.0, 0.0, 0.0])
    gyro  = imu.get('gyro',  [0.0, 0.0, 0.0])
    grav  = imu.get('grav',  [0.0, 0.0, 0.0])
    roll_deg  = imu.get('roll_deg',  0.0)
    pitch_deg = imu.get('pitch_deg', 0.0)

    interp_cruda    = frame.get('interpolada_cruda', [0.0] * 450)
    interp_nivelada = frame.get('interpolada',        [0.0] * 450)

    ts = datetime.datetime.now().isoformat(timespec='milliseconds')

    # ---------- Archivo 1: LiDAR crudo interpolado + IMU en vivo + etiquetas ----------
    # Orden fijo: dist_0..dist_449, IMU (accel, gyro, grav, roll, pitch),
    # binarios (sup_frontal, sup_trasera, esc_frontal, esc_trasera, obstaculo),
    # floats (ang_sup_frontal, ang_sup_trasera, altura_escalones, dist_obstaculo).
    # Sin columna de timestamp.
    fila_crudo = (
        list(interp_cruda)
        + list(accel) + list(gyro) + list(grav)
        + [roll_deg, pitch_deg]
        + [
            etiq.get('sup_frontal', 0),
            etiq.get('sup_trasera', 0),
            etiq.get('esc_frontal', 0),
            etiq.get('esc_trasera', 0),
            etiq.get('obstaculo',   0),
        ]
        + [
            etiq.get('ang_frontal', 0.5),
            etiq.get('ang_trasero', 0.5),
            etiq.get('altura_esc',  0.0),
            etiq.get('dist_obs',    0.0),
        ]
    )
    ya_existe_crudo = os.path.exists(CSV_CRUDO_PATH)
    with open(CSV_CRUDO_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        if not ya_existe_crudo:
            header = (
                [f'dist_{i}' for i in range(450)]
                + ['imu_accel_x', 'imu_accel_y', 'imu_accel_z']
                + ['imu_gyro_x',  'imu_gyro_y',  'imu_gyro_z']
                + ['imu_grav_x',  'imu_grav_y',  'imu_grav_z']
                + ['imu_roll_deg', 'imu_pitch_deg']
                + ['sup_frontal', 'sup_trasera', 'esc_frontal', 'esc_trasera', 'obstaculo']
                + ['ang_sup_frontal', 'ang_sup_trasera', 'altura_escalones', 'dist_obstaculo']
            )
            w.writerow(header)
        w.writerow(fila_crudo)

    # ---------- Archivo 2: LiDAR nivelado interpolado + etiquetas ----------
    fila_nivelado = (
        [ts]
        + list(interp_nivelada)
        + [
            etiq.get('sup_frontal', 0),
            etiq.get('ang_frontal', 0.5),
            etiq.get('sup_trasera', 0),
            etiq.get('ang_trasero', 0.5),
            etiq.get('esc_frontal', 0),
            etiq.get('esc_trasera', 0),
            etiq.get('altura_esc',  0.0),
            etiq.get('obstaculo',   0),
            etiq.get('dist_obs',    0.0),
        ]
    )
    ya_existe_nivelado = os.path.exists(CSV_NIVELADO_PATH)
    with open(CSV_NIVELADO_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        if not ya_existe_nivelado:
            header = (
                ['timestamp']
                + [f'lidar_nivelado_{i}' for i in range(450)]
                + ['sup_frontal', 'ang_frontal',
                   'sup_trasera', 'ang_trasero',
                   'esc_frontal', 'esc_trasera',
                   'altura_esc',  'obstaculo',  'dist_obs']
            )
            w.writerow(header)
        w.writerow(fila_nivelado)

    filas_guardadas += 1
    return jsonify({'status': 'ok', 'total_filas': filas_guardadas})


# ==========================================
# PÁGINA HTML
# ==========================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LIDAR+PROTESIS 2026 · LiDAR + IMU Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:     #050e1a;  --panel:  #08152a;  --border: #0f2340;
  --accent: #00ffa3;  --amber:  #f59e0b;  --blue:   #3b82f6;
  --violet: #a78bfa;  --text:   #d4e4f7;  --muted:  #4a6887;
  --danger: #ef4444;
}
body { background:var(--bg); color:var(--text); font-family:'Space Grotesk',sans-serif; height:100vh; display:flex; flex-direction:column; overflow:hidden; }

/* HEADER */
header { display:flex; align-items:center; justify-content:space-between; padding:9px 20px; border-bottom:1px solid var(--border); background:var(--panel); flex-shrink:0; }
.h-brand { display:flex; align-items:center; gap:14px; }
.h-name  { font-size:12px; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); }
.h-sub   { font-size:11px; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.h-right { display:flex; align-items:center; gap:22px; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); }
.h-stat  { display:flex; align-items:center; gap:6px; }
.sdot { width:8px; height:8px; border-radius:50%; background:var(--danger); transition:background .3s, box-shadow .3s; }
.sdot.on { background:var(--accent); box-shadow:0 0 7px var(--accent); }

/* MAIN */
main { flex:1; display:grid; grid-template-columns:1fr 1fr 1fr 260px; overflow:hidden; min-height:0; }

/* CHART CELLS */
.chart-cell { display:flex; flex-direction:column; align-items:center; justify-content:space-between; padding:10px 10px 8px; border-right:1px solid var(--border); overflow:hidden; gap:6px; }
.chart-cell canvas { display:block; flex-shrink:0; }
.chart-hdr { width:100%; display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
.chart-title { font-size:9px; font-weight:600; letter-spacing:.22em; text-transform:uppercase; color:var(--muted); }
.chart-badge { font-family:'JetBrains Mono',monospace; font-size:9px; padding:1px 6px; border-radius:3px; border:1px solid; }

/* ZOOM */
.zoom-row { width:100%; display:flex; align-items:center; gap:8px; flex-shrink:0; padding:0 2px; }
.zoom-lbl { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; flex-shrink:0; }
.zoom-val { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; color:var(--accent); width:48px; text-align:right; flex-shrink:0; }
input[type="range"] { -webkit-appearance:none; appearance:none; flex:1; height:3px; border-radius:2px; background:var(--border); outline:none; cursor:pointer; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance:none; width:13px; height:13px; border-radius:50%; cursor:pointer; }
input[type="range"]::-moz-range-thumb    { width:13px; height:13px; border-radius:50%; border:none; cursor:pointer; }
#z-cruda::-webkit-slider-thumb    { background:var(--blue); }
#z-nivelada::-webkit-slider-thumb { background:var(--accent); }
#z-interp::-webkit-slider-thumb   { background:var(--amber); }
#z-cruda::-moz-range-thumb    { background:var(--blue); }
#z-nivelada::-moz-range-thumb { background:var(--accent); }
#z-interp::-moz-range-thumb   { background:var(--amber); }

/* SIDEBAR */
.sidebar { display:flex; flex-direction:column; background:var(--panel); overflow-y:auto; }

/* SECCIÓN GENÉRICA */
.s-sec { padding:11px 13px; border-bottom:1px solid var(--border); }
.sec-title { font-size:9px; font-weight:600; letter-spacing:.26em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }

/* IMU ROWS */
.imu-row  { display:flex; align-items:center; gap:6px; margin-bottom:5px; }
.imu-axis { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); width:12px; flex-shrink:0; }
.bar-wrap { flex:1; height:3px; background:var(--border); border-radius:2px; overflow:hidden; }
.bar-fill { height:100%; border-radius:2px; transition:width .08s linear; }
.imu-val  { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text); width:58px; text-align:right; flex-shrink:0; }

/* GRAVEDAD */
.grav-wrap { display:flex; flex-direction:column; align-items:center; gap:9px; }
#grav-canvas { display:block; }
.grav-stats { width:100%; display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.gstat { background:var(--border); border-radius:4px; padding:5px 8px; }
.gstat-lbl { font-size:9px; color:var(--muted); font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:.1em; margin-bottom:1px; }
.gstat-val { font-size:16px; font-weight:600; font-family:'JetBrains Mono',monospace; color:var(--amber); }

/* ── ETIQUETAS ─────────────────────────────── */
.lbl-hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:9px; }
.lbl-count { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--accent); font-weight:600; }

/* Binarios: grid 2 columnas */
.bin-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px; margin-bottom:10px; }
.bin-item { display:flex; flex-direction:column; gap:3px; }
.bin-name { font-family:'JetBrains Mono',monospace; font-size:8px; color:var(--muted); letter-spacing:.04em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
/* El obstáculo ocupa ambas columnas, centrado y destacado */
.bin-item.full { grid-column:1 / -1; flex-direction:column; align-items:center; justify-content:center; gap:7px; margin-top:6px; }
.bin-item.full .bin-name { font-size:11px; letter-spacing:.08em; }
.tog {
  padding:5px 0; background:var(--border);
  border:1px solid var(--muted); color:var(--muted);
  font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:600;
  border-radius:4px; cursor:pointer; transition:all .12s; width:100%;
}
.tog.on { background:rgba(0,255,163,0.14); border-color:var(--accent); color:var(--accent); box-shadow:0 0 8px rgba(0,255,163,0.2); }
.full .tog { width:72%; max-width:170px; padding:14px 0; font-size:26px; }

/* Floats */
.ff { margin-bottom:8px; }
.ff-hdr { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px; }
.ff-name   { font-family:'JetBrains Mono',monospace; font-size:8px; color:var(--muted); letter-spacing:.04em; }
.ff-stored { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--amber); }
.ff-ctrl { display:flex; align-items:center; gap:5px; }
.ff-ctrl input[type="number"] {
  flex:1; background:var(--border); border:1px solid var(--muted);
  color:var(--text); font-family:'JetBrains Mono',monospace; font-size:13px;
  padding:5px 7px; border-radius:4px; text-align:right;
  -moz-appearance:textfield;
}
.ff-ctrl input[type="number"]::-webkit-outer-spin-button,
.ff-ctrl input[type="number"]::-webkit-inner-spin-button { -webkit-appearance:none; }
.ff-ctrl input[type="number"]:focus { outline:none; border-color:var(--accent); }
.ff-unit { font-size:9px; color:var(--muted); font-family:'JetBrains Mono',monospace; flex-shrink:0; }

/* Botón guardar */
.save-btn {
  width:100%; padding:9px; margin-top:4px;
  background:rgba(0,255,163,0.10); border:1px solid var(--accent);
  color:var(--accent); font-family:'Space Grotesk',sans-serif;
  font-size:12px; font-weight:600; letter-spacing:.07em;
  border-radius:4px; cursor:pointer; transition:all .13s;
}
.save-btn:hover:not(:disabled) { background:rgba(0,255,163,0.2); }
.save-btn:active:not(:disabled) { transform:scale(0.98); }
.save-btn:disabled { opacity:.4; cursor:not-allowed; }
.save-status { font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--muted); text-align:center; margin-top:5px; min-height:14px; }
</style>
</head>
<body>

<header>
  <div class="h-brand">
    <span class="h-name">LIDAR+PROTESIS 2026</span>
    <span class="h-sub">LiDAR + IMU · Tiempo Real</span>
  </div>
  <div class="h-right">
    <div class="h-stat"><div class="sdot" id="sdot"></div><span id="stext">Desconectado</span></div>
    <div class="h-stat"><span id="fps-ctr">-- fps</span></div>
    <div class="h-stat"><span id="pts-ctr">-- pts</span></div>
  </div>
</header>

<main>

  <!-- CANVAS 1: Cruda -->
  <section class="chart-cell">
    <div class="chart-hdr">
      <span class="chart-title">Vuelta Cruda</span>
      <span class="chart-badge" style="color:var(--blue);border-color:var(--blue)">RAW</span>
    </div>
    <canvas id="c-cruda"></canvas>
    <div class="zoom-row">
      <span class="zoom-lbl">Zoom</span>
      <input type="range" id="z-cruda"   min="500" max="12000" step="500" value="12000" oninput="onZoom('cruda',this.value)">
      <span class="zoom-val" id="zv-cruda">12.0 m</span>
    </div>
  </section>

  <!-- CANVAS 2: Nivelada -->
  <section class="chart-cell">
    <div class="chart-hdr">
      <span class="chart-title">Vuelta Nivelada</span>
      <span class="chart-badge" style="color:var(--accent);border-color:var(--accent)">IMU</span>
    </div>
    <canvas id="c-nivelada"></canvas>
    <div class="zoom-row">
      <span class="zoom-lbl">Zoom</span>
      <input type="range" id="z-nivelada" min="500" max="12000" step="500" value="12000" oninput="onZoom('nivelada',this.value)">
      <span class="zoom-val" id="zv-nivelada">12.0 m</span>
    </div>
  </section>

  <!-- CANVAS 3: Interpolada -->
  <section class="chart-cell">
    <div class="chart-hdr">
      <span class="chart-title">Interpolada</span>
      <span class="chart-badge" style="color:var(--amber);border-color:var(--amber)">450 pts</span>
    </div>
    <canvas id="c-interp"></canvas>
    <div class="zoom-row">
      <span class="zoom-lbl">Zoom</span>
      <input type="range" id="z-interp"  min="500" max="12000" step="500" value="12000" oninput="onZoom('interp',this.value)">
      <span class="zoom-val" id="zv-interp">12.0 m</span>
    </div>
  </section>

  <!-- SIDEBAR -->
  <aside class="sidebar">

    <!-- ── ETIQUETAS ── -->
    <div class="s-sec">
      <div class="lbl-hdr">
        <div class="sec-title" style="margin-bottom:0">Etiquetas</div>
        <span class="lbl-count"><span id="frame-count">0</span> frames</span>
      </div>

      <!-- Binarios -->
      <div class="bin-grid">
        <div class="bin-item">
          <span class="bin-name">Sup. Frontal</span>
          <button class="tog" id="tog-sf" onclick="togBin('sf','sup_frontal')">0</button>
        </div>
        <div class="bin-item">
          <span class="bin-name">Sup. Trasera</span>
          <button class="tog" id="tog-st" onclick="togBin('st','sup_trasera')">0</button>
        </div>
        <div class="bin-item">
          <span class="bin-name">Esc. Frontal</span>
          <button class="tog" id="tog-ef" onclick="togBin('ef','esc_frontal')">0</button>
        </div>
        <div class="bin-item">
          <span class="bin-name">Esc. Trasera</span>
          <button class="tog" id="tog-et" onclick="togBin('et','esc_trasera')">0</button>
        </div>
        <div class="bin-item full">
          <span class="bin-name">Obstáculo</span>
          <button class="tog" id="tog-ob" onclick="togBin('ob','obstaculo')">0</button>
        </div>
      </div>

      <!-- Floats -->
      <div class="ff">
        <div class="ff-hdr">
          <span class="ff-name">Ángulo Frontal</span>
          <span class="ff-stored" id="st-af">0.500</span>
        </div>
        <div class="ff-ctrl">
          <input type="number" id="n-af" step="0.1" value="0"  oninput="syncFloat('af',this.value)">
          <span class="ff-unit">°</span>
        </div>
      </div>

      <div class="ff">
        <div class="ff-hdr">
          <span class="ff-name">Ángulo Trasero</span>
          <span class="ff-stored" id="st-at">0.500</span>
        </div>
        <div class="ff-ctrl">
          <input type="number" id="n-at" step="0.1" value="0"  oninput="syncFloat('at',this.value)">
          <span class="ff-unit">°</span>
        </div>
      </div>

      <div class="ff">
        <div class="ff-hdr">
          <span class="ff-name">Altura Escalones</span>
          <span class="ff-stored" id="st-ae">0.000</span>
        </div>
        <div class="ff-ctrl">
          <input type="number" id="n-ae" step="0.1" value="0"   oninput="syncFloat('ae',this.value)">
          <span class="ff-unit">cm</span>
        </div>
      </div>

      <div class="ff">
        <div class="ff-hdr">
          <span class="ff-name">Dist. Obstáculo</span>
          <span class="ff-stored" id="st-do">0.000</span>
        </div>
        <div class="ff-ctrl">
          <input type="number" id="n-do" step="0.1" value="30"  oninput="syncFloat('do',this.value)">
          <span class="ff-unit">cm</span>
        </div>
      </div>

      <button class="save-btn" id="save-btn" onclick="guardarFrame()">
        ▼ Guardar Frame &nbsp;<kbd style="font-size:9px;opacity:.6">[G]</kbd>
      </button>
      <div class="save-status" id="save-status">Sin datos LiDAR</div>
    </div>

    <!-- ── ACELERÓMETRO ── -->
    <div class="s-sec">
      <div class="sec-title">Acelerómetro (m/s²)</div>
      <div class="imu-row"><span class="imu-axis">X</span><div class="bar-wrap"><div class="bar-fill" id="ax-b" style="background:var(--blue);width:50%"></div></div><span class="imu-val" id="ax-v">0.000</span></div>
      <div class="imu-row"><span class="imu-axis">Y</span><div class="bar-wrap"><div class="bar-fill" id="ay-b" style="background:var(--blue);width:50%"></div></div><span class="imu-val" id="ay-v">0.000</span></div>
      <div class="imu-row"><span class="imu-axis">Z</span><div class="bar-wrap"><div class="bar-fill" id="az-b" style="background:var(--blue);width:50%"></div></div><span class="imu-val" id="az-v">0.000</span></div>
    </div>

    <!-- ── GIROSCOPIO ── -->
    <div class="s-sec">
      <div class="sec-title">Giroscopio (rad/s)</div>
      <div class="imu-row"><span class="imu-axis">X</span><div class="bar-wrap"><div class="bar-fill" id="gx-b" style="background:var(--violet);width:50%"></div></div><span class="imu-val" id="gx-v">0.000</span></div>
      <div class="imu-row"><span class="imu-axis">Y</span><div class="bar-wrap"><div class="bar-fill" id="gy-b" style="background:var(--violet);width:50%"></div></div><span class="imu-val" id="gy-v">0.000</span></div>
      <div class="imu-row"><span class="imu-axis">Z</span><div class="bar-wrap"><div class="bar-fill" id="gz-b" style="background:var(--violet);width:50%"></div></div><span class="imu-val" id="gz-v">0.000</span></div>
    </div>

    <!-- ── GRAVEDAD ── -->
    <div class="s-sec">
      <div class="sec-title">Vector de Gravedad</div>
      <div class="grav-wrap">
        <canvas id="grav-canvas" width="200" height="200"></canvas>
        <div class="grav-stats">
          <div class="gstat"><div class="gstat-lbl">Ángulo XY</div><div class="gstat-val" id="g-angle">--°</div></div>
          <div class="gstat"><div class="gstat-lbl">Desfase</div><div class="gstat-val"  id="g-desfase">--°</div></div>
        </div>
      </div>
    </div>

  </aside>
</main>

<script>
// ── Constantes ─────────────────────────────────────────────────────────────
const MAX_MM     = 12000;
const ANGLE_STEP = 0.8;
const NUM_PTS    = 450;

// ── Zoom por canvas ─────────────────────────────────────────────────────────
const zoom = { cruda:12000, nivelada:12000, interp:12000 };

// ── Cache del último frame ──────────────────────────────────────────────────
let lastFrame = null;

// ─── CONFIGURACIÓN DE ETIQUETAS ────────────────────────────────────────────
// Cada campo float tiene: la clave JSON que se enviará al backend,
// los límites de display, y el valor inicial de display.
const FLOAT_CFG = {
  af: { key:'ang_frontal', min:-15, max:15,  unit:'°',  init:0  },
  at: { key:'ang_trasero', min:-15, max:15,  unit:'°',  init:0  },
  ae: { key:'altura_esc',  min:0,   max:30,  unit:'cm', init:0  },
  do: { key:'dist_obs',    min:30,  max:300, unit:'cm', init:30 },
};

// Estado actual de las etiquetas (valores almacenados 0-1 o 0/1)
const labels = {
  sup_frontal: 0,  ang_frontal: 0.5,
  sup_trasera: 0,  ang_trasero: 0.5,
  esc_frontal: 0,  esc_trasera: 0,
  altura_esc:  0,
  obstaculo:   0,
  dist_obs:    0,
};

let framesSaved = 0;

// ── Toggle binario ─────────────────────────────────────────────────────────
function togBin(id, key) {
  labels[key] = labels[key] === 0 ? 1 : 0;
  const btn = document.getElementById('tog-' + id);
  btn.textContent = labels[key];
  btn.classList.toggle('on', labels[key] === 1);
}

// ── Sync number input y calcula valor almacenado (saturado 0-1) ────────────
// No tocamos el valor que el usuario está escribiendo (permite negativos,
// valores fuera de rango, etc.). Solo el valor NORMALIZADO que se guarda
// se satura entre 0 y 1.
function syncFloat(id, val) {
  const cfg    = FLOAT_CFG[id];
  const parsed = parseFloat(val);
  if (isNaN(parsed)) return; // el usuario está a mitad de escribir (ej. "-"), no hacemos nada aún
  const stored = Math.max(0, Math.min(1, (parsed - cfg.min) / (cfg.max - cfg.min)));
  labels[cfg.key] = stored;
  document.getElementById('st-' + id).textContent = stored.toFixed(3);
}

// Inicializa los displays a sus valores por defecto
function initLabels() {
  Object.keys(FLOAT_CFG).forEach(id => syncFloat(id, FLOAT_CFG[id].init));
}

// ── Guardar frame ──────────────────────────────────────────────────────────
async function guardarFrame() {
  if (!lastFrame) {
    document.getElementById('save-status').textContent = '⚠ Sin datos LiDAR todavía';
    return;
  }
  const btn = document.getElementById('save-btn');
  btn.disabled = true;
  try {
    const resp = await fetch('/api/guardar', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame: lastFrame, labels })
    });
    const d = await resp.json();
    framesSaved++;
    document.getElementById('frame-count').textContent = framesSaved;
    document.getElementById('save-status').textContent =
      '✓ ' + new Date().toLocaleTimeString('es-CL') + '  —  total ' + d.total_filas;
  } catch (e) {
    document.getElementById('save-status').textContent = '✗ Error de conexión';
  }
  btn.disabled = false;
}

// Atajo de teclado: G para guardar (solo cuando no hay un input activo)
document.addEventListener('keydown', e => {
  if ((e.key === 'g' || e.key === 'G') && !e.target.matches('input,textarea,button,select')) {
    e.preventDefault();
    guardarFrame();
  }
});

// ── Helpers ────────────────────────────────────────────────────────────────
function fmtDist(mm) { return mm >= 1000 ? (mm/1000).toFixed(1)+' m' : mm+' mm'; }
function rings(maxR) { return [0.25, 0.5, 0.75, 1.0].map(f => maxR * f); }
function barPct(v, mv) { return Math.max(0, Math.min(100, 50 + (v/mv)*50)) + '%'; }

// ── Resize de los 3 canvas ──────────────────────────────────────────────────
function resizeAll() {
  ['c-cruda','c-nivelada','c-interp'].forEach(id => {
    const c    = document.getElementById(id);
    const cell = c.closest('.chart-cell');
    const avW  = cell.clientWidth  - 20;
    const avH  = cell.clientHeight - 62;
    const size = Math.max(Math.min(avW, avH), 80);
    c.width = c.height = size;
  });
  if (lastFrame) redrawAll(lastFrame);
}
window.addEventListener('resize', resizeAll);

// ── Grilla polar compartida ────────────────────────────────────────────────
function drawGrid(ctx, cx, cy, maxR, maxRange) {
  ctx.beginPath(); ctx.arc(cx, cy, maxR, 0, 2*Math.PI);
  ctx.fillStyle = '#020a14'; ctx.fill();

  rings(maxRange).forEach(r => {
    const pr = (r/maxRange)*maxR;
    ctx.beginPath(); ctx.arc(cx, cy, pr, 0, 2*Math.PI);
    ctx.strokeStyle = 'rgba(59,130,246,0.16)'; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = 'rgba(74,104,135,0.7)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(fmtDist(r), cx + pr + 3, cy - 3);
  });

  for (let d = 0; d < 360; d += 30) {
    const rad = d * Math.PI / 180;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + maxR*Math.cos(rad), cy + maxR*Math.sin(rad));
    ctx.strokeStyle = 'rgba(59,130,246,0.09)'; ctx.lineWidth = 1; ctx.stroke();
    if (d % 90 === 0) {
      const lx = cx + (maxR+12)*Math.cos(rad), ly = cy + (maxR+12)*Math.sin(rad);
      ctx.fillStyle = 'rgba(74,104,135,0.75)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(d + '°', lx, ly);
    }
  }
  ctx.strokeStyle = 'rgba(245,158,11,0.6)'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx-5,cy); ctx.lineTo(cx+5,cy);
  ctx.moveTo(cx,cy-5); ctx.lineTo(cx,cy+5);
  ctx.stroke();
}

// ── Canvas 1: Cruda ────────────────────────────────────────────────────────
function drawCruda(points, maxRange) {
  const c = document.getElementById('c-cruda'), ctx = c.getContext('2d');
  const cx = c.width/2, maxR = cx*0.86, sc = maxR/maxRange;
  ctx.clearRect(0,0,c.width,c.width);
  drawGrid(ctx, cx, cx, maxR, maxRange);
  ctx.fillStyle = '#3b82f6';
  let vis = 0;
  for (const [ang, dist] of points) {
    if (dist <= 0 || dist > maxRange) continue;
    const rad = ang * Math.PI / 180;
    ctx.beginPath(); ctx.arc(cx + dist*sc*Math.cos(rad), cx + dist*sc*Math.sin(rad), 1.5, 0, 2*Math.PI); ctx.fill();
    vis++;
  }
  ctx.fillStyle='rgba(74,104,135,0.7)'; ctx.font='9px JetBrains Mono,monospace';
  ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText(vis+' pts',6,6);
}

// ── Canvas 2: Nivelada ─────────────────────────────────────────────────────
function drawNivelada(points, maxRange) {
  const c = document.getElementById('c-nivelada'), ctx = c.getContext('2d');
  const cx = c.width/2, maxR = cx*0.86, sc = maxR/maxRange;
  ctx.clearRect(0,0,c.width,c.width);
  drawGrid(ctx, cx, cx, maxR, maxRange);
  let vis = 0;
  for (const [ang, dist] of points) {
    if (dist <= 0 || dist > maxRange) continue;
    const rad = ang * Math.PI / 180;
    const bright = Math.max(0.3, 1 - dist/maxRange);
    ctx.fillStyle = 'rgba(0,255,163,' + bright + ')';
    ctx.beginPath(); ctx.arc(cx + dist*sc*Math.cos(rad), cx + dist*sc*Math.sin(rad), 1.5, 0, 2*Math.PI); ctx.fill();
    vis++;
  }
  ctx.fillStyle='rgba(74,104,135,0.7)'; ctx.font='9px JetBrains Mono,monospace';
  ctx.textAlign='left'; ctx.textBaseline='top'; ctx.fillText(vis+' pts',6,6);
}

// ── Canvas 3: Interpolada ──────────────────────────────────────────────────
function drawInterp(points, maxRange) {
  const c = document.getElementById('c-interp'), ctx = c.getContext('2d');
  const cx = c.width/2, maxR = cx*0.86, sc = maxR/maxRange;
  ctx.clearRect(0,0,c.width,c.width);
  drawGrid(ctx, cx, cx, maxR, maxRange);
  if (!points || points.length !== NUM_PTS) return;
  ctx.beginPath();
  for (let i = 0; i < NUM_PTS; i++) {
    const dist = Math.min(points[i], maxRange);
    const rad  = (i * ANGLE_STEP) * Math.PI / 180;
    const px = cx + dist*sc*Math.cos(rad), py = cx + dist*sc*Math.sin(rad);
    i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.closePath();
  ctx.fillStyle='rgba(245,158,11,0.07)'; ctx.fill();
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=1.5; ctx.stroke();
  const out = points.filter(d => d > maxRange).length;
  if (out > 0) {
    ctx.fillStyle='rgba(245,158,11,0.55)'; ctx.font='9px JetBrains Mono,monospace';
    ctx.textAlign='left'; ctx.textBaseline='top';
    ctx.fillText('>'+fmtDist(maxRange)+': '+out+' pts', 6, 6);
  }
}

// ── Canvas Gravedad ────────────────────────────────────────────────────────
function drawGrav(gx, gy) {
  const c = document.getElementById('grav-canvas'), ctx = c.getContext('2d');
  const cx = c.width/2, R = cx*0.80;
  ctx.clearRect(0,0,c.width,c.width);
  ctx.beginPath(); ctx.arc(cx,cx,R,0,2*Math.PI);
  ctx.fillStyle='#020a14'; ctx.fill();
  ctx.strokeStyle='#0f2340'; ctx.lineWidth=1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx,cx,R*0.12,0,2*Math.PI);
  ctx.strokeStyle='rgba(245,158,11,0.2)'; ctx.lineWidth=1; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,cx); ctx.lineTo(cx,cx+R*0.92);
  ctx.strokeStyle='rgba(245,158,11,0.18)'; ctx.setLineDash([3,5]); ctx.lineWidth=1; ctx.stroke();
  ctx.setLineDash([]);
  const mag = Math.hypot(gx,gy)||1;
  const ax = cx + R*0.88*(gx/mag), ay = cx - R*0.88*(gy/mag);
  ctx.beginPath(); ctx.moveTo(cx,cx); ctx.lineTo(ax,ay);
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2.2; ctx.stroke();
  const ang = Math.atan2(ay-cx, ax-cx), HL=9;
  ctx.beginPath();
  ctx.moveTo(ax,ay); ctx.lineTo(ax-HL*Math.cos(ang-.5), ay-HL*Math.sin(ang-.5));
  ctx.moveTo(ax,ay); ctx.lineTo(ax-HL*Math.cos(ang+.5), ay-HL*Math.sin(ang+.5));
  ctx.strokeStyle='#f59e0b'; ctx.lineWidth=2; ctx.stroke();
  ctx.beginPath(); ctx.arc(ax,ay,3,0,2*Math.PI); ctx.fillStyle='#f59e0b'; ctx.fill();
  ctx.fillStyle='rgba(74,104,135,0.55)'; ctx.font='9px JetBrains Mono,monospace';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('+X',cx+R+10,cx); ctx.fillText('-X',cx-R-10,cx);
  ctx.fillText('+Y',cx,cx-R-8); ctx.fillText('-Y',cx,cx+R+8);
  const angDeg = Math.atan2(gy,gx)*180/Math.PI;
  document.getElementById('g-angle').textContent   = angDeg.toFixed(1)+'°';
  document.getElementById('g-desfase').textContent = (angDeg+90).toFixed(1)+'°';
}

// ── IMU panel ──────────────────────────────────────────────────────────────
function updateIMU(imu) {
  const [ax,ay,az]=imu.accel, [gx,gy,gz]=imu.gyro, [vx,vy]=imu.grav;
  document.getElementById('ax-v').textContent=ax.toFixed(3); document.getElementById('ax-b').style.width=barPct(ax,15);
  document.getElementById('ay-v').textContent=ay.toFixed(3); document.getElementById('ay-b').style.width=barPct(ay,15);
  document.getElementById('az-v').textContent=az.toFixed(3); document.getElementById('az-b').style.width=barPct(az,15);
  document.getElementById('gx-v').textContent=gx.toFixed(3); document.getElementById('gx-b').style.width=barPct(gx,5);
  document.getElementById('gy-v').textContent=gy.toFixed(3); document.getElementById('gy-b').style.width=barPct(gy,5);
  document.getElementById('gz-v').textContent=gz.toFixed(3); document.getElementById('gz-b').style.width=barPct(gz,5);
  drawGrav(vx, vy);
}

// ── Zoom ───────────────────────────────────────────────────────────────────
function onZoom(key, val) {
  zoom[key] = parseInt(val);
  document.getElementById('zv-'+key).textContent = fmtDist(zoom[key]);
  if (lastFrame) redrawAll(lastFrame);
}

// ── Redibujo completo ──────────────────────────────────────────────────────
function redrawAll(d) {
  drawCruda(d.cruda, zoom.cruda);
  drawNivelada(d.nivelada, zoom.nivelada);
  drawInterp(d.interpolada, zoom.interp);
  updateIMU(d.imu);
}

// ── FPS ────────────────────────────────────────────────────────────────────
let frames=0, lastFpsTs=performance.now();
function tickFps() {
  frames++;
  const now = performance.now();
  if (now - lastFpsTs >= 1000) {
    document.getElementById('fps-ctr').textContent = frames+' fps';
    frames=0; lastFpsTs=now;
  }
}

// ── SSE ────────────────────────────────────────────────────────────────────
let es = null;
function conectar() {
  if (es) es.close();
  es = new EventSource('/stream');
  es.onopen = () => {
    document.getElementById('sdot').classList.add('on');
    document.getElementById('stext').textContent = 'Conectado';
    document.getElementById('save-status').textContent = 'Listo para guardar';
  };
  es.onmessage = e => {
    try {
      const d = JSON.parse(e.data);
      lastFrame = d;
      redrawAll(d);
      document.getElementById('pts-ctr').textContent = d.cruda.length+' pts';
      tickFps();
    } catch(err) { console.error('[SSE]', err); }
  };
  es.onerror = () => {
    document.getElementById('sdot').classList.remove('on');
    document.getElementById('stext').textContent = 'Reconectando...';
    es.close(); setTimeout(conectar, 2000);
  };
}

resizeAll();
initLabels();
conectar();
</script>
</body>
</html>"""


# ==========================================
# PUNTO DE ENTRADA
# ==========================================

def lanzar_servidor_web(cola_entrante):
    os.sched_setaffinity(0, {0})
    global cola_global
    cola_global = cola_entrante
    print("[WEB] Servidor iniciado en http://0.0.0.0:5500")
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5500, debug=False, use_reloader=False, threaded=True)