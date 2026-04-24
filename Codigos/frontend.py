from supabase import create_client, Client
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys

# CONFIG

SUPABASE_URL = "https://ggrpykvwguknefqbhlvx.supabase.co"
SUPABASE_KEY = "sb_publishable_AoTEVrhfuEzJVbHpgpbIQA_RQ2YdIqB"
NOMBRE_TABLA = "Datos y Predicciones"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ultimo_id = None

def obtener_ultimo_dato():
    try:
        response = (
            supabase
            .table(NOMBRE_TABLA)
            .select("*")
            .order("id", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as e:
        print(f"Error consultando DB: {e}")

    return None

class LidarViewer(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("LiDAR en tiempo real")
        self.resize(800, 600)

        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)

        # Opcional: mejorar visual
        self.plot.showGrid(x=True, y=True)

        self.scatter = pg.ScatterPlotItem(size=5)
        self.plot.addItem(self.scatter)

        # Timer para actualizar
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # 1 Hz

    def update_data(self):
        global ultimo_id

        dato = obtener_ultimo_dato()

        if not dato:
            return

        if dato["id"] == ultimo_id:
            return

        ultimo_id = dato["id"]

        clase = dato.get("clase", "Desconocida")

        theta = [float(x) for x in dato["lista_theta"]]
        r = [float(x) for x in dato["lista_r"]]

        theta = np.array(theta)
        r = np.array(r)

        theta_rad = np.radians(theta)

        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)

        y = y + 0.5

        puntos = [{'pos': (x[i], y[i])} for i in range(len(x))]

        self.scatter.setData(puntos)

        self.plot.setTitle(f"Clase predicha: {clase}")

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    viewer = LidarViewer()
    viewer.show()
    sys.exit(app.exec_())
