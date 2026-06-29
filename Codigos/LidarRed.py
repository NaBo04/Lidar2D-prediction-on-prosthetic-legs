import torch
import torch.nn as nn

class LidarCNN(nn.Module):
    def __init__(self):
        super(LidarCNN, self).__init__()
        # Entrada: [Batch, 1, 450]
        self.lidar_red = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3), # [Batch, 16, 225] aprx
            nn.BatchNorm1d(16), # normalizamos para las 16 dimensiones
            nn.ReLU(), #capita relu
            nn.MaxPool1d(kernel_size=2), # Pool global nos quedan [Batch, 16, 112] maomeno despues printeo
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2), # [Batch, 32, 56 
            nn.BatchNorm1d(32), #lo mismo, normalizamos para 32 canales para todos los batches
            nn.ReLU(), #sexo relu
            nn.MaxPool1d(kernel_size=2), # Redimension aproz a [Batch, 32, 28]
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8) # Forzamos a que el tamaño final sea fijo (64 canales x 8 features)
        )
        
        # Rama donde juntamos todo
        # Tamaño de entrada = (64 * 8 del LiDAR) = 512 
        self.fc_lidar = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2), #este de aca es para no sobre entrenars
            nn.Linear(128, 9) # Salida lineal de 9 neuronas
        )

    def forward(self, lidar):
        if lidar.dim() == 2:
            #si viene [Batch, 1, 450] no pasa nada tiene 3 dim
            #si viene [Batch, 450] agrega una dimension en la posicion 1:
            lidar = lidar.unsqueeze(1)
            
        # LiDAR
        x_lidar = self.lidar_red(lidar)
        x_lidar = torch.flatten(x_lidar, start_dim=1) # Convierte a [B, 512] para juntar todo luego
        
        # Capas final donde se junta todo
        output = self.fc_lidar(x_lidar) # [B, 9]
        
        return output