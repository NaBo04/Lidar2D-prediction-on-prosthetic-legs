import torch
import torch.nn as nn

class LidarCNN(nn.Module):
    def __init__(self):
        super(LidarCNN, self).__init__()
        # Entrada: [Batch, 1, 450]
        self.lidar_red = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=3),
            nn.BatchNorm1d(16), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2), 
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(32), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2), 
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # CAMBIO: Reducimos de 8 a 4 para bajar la cantidad de parámetros al aplanar
            nn.AdaptiveAvgPool1d(4) 
        )
        
        # Rama donde juntamos todo
        # Tamaño de entrada = (64 * 4 del LiDAR) = 256 
        self.fc_lidar = nn.Sequential(
            # CAMBIO: Dropout agresivo justo después del aplanado
            nn.Dropout(p=0.3), 
            nn.Linear(256, 64), # CAMBIO: Reducimos las neuronas ocultas de 128 a 64
            nn.BatchNorm1d(64), # CAMBIO: Un BatchNorm extra aquí ayuda mucho a regularizar
            nn.ReLU(),
            nn.Dropout(p=0.5), # CAMBIO: Aumentamos el Dropout final de 0.2 a 0.5
            # CAMBIO: Aumentamos el Dropout final de 0.2 a 0.5
            nn.Linear(64, 5) # Salida lineal de 5 neuronas
        )

    def forward(self, lidar):
        if lidar.dim() == 2:
            lidar = lidar.unsqueeze(1)
            
        x_lidar = self.lidar_red(lidar)
        x_lidar = torch.flatten(x_lidar, start_dim=1) 
        
        output = self.fc_lidar(x_lidar) 
        
        return output