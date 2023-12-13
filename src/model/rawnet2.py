import torch.nn as nn
import torch.nn.functional as F
import torch

from src.base import BaseModel

from src.model.layers import SincConv


NEGATIVE_SLOPE = 0.3


class FMS(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.lin_layer = nn.Linear(num_filters, num_filters)

    def forward(self, x):
        # x - (B, F, T)

        s = F.sigmoid(self.lin_layer(self.avg_pool(x).transpose(1, 2))).transpose(1, 2)   # (B, F, 1)

        return x * s + s

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.fms = FMS(out_channels)

        if in_channels != out_channels:
            self.upsample_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
            self.upsample = True
        else:
            self.upsample = False


    def forward(self, x):
        # (B, F, T)

        y = F.leaky_relu(self.bn1(x), negative_slope=NEGATIVE_SLOPE)
        y = self.conv1(y)
        y = F.leaky_relu(self.bn2(y), negative_slope=NEGATIVE_SLOPE)
        y = self.conv2(y)
        if self.upsample:
            z = y + self.upsample_conv(x)
        else:
            z = y + x
            
        z = F.max_pool1d(z, 3)
        z = self.fms(z)
        return z





class RawNet2(BaseModel):
    def __init__(self, sinc_filter_length, hidden_channels, gru_dim, **batch):
        super().__init__(**batch)

        channel1, channel2 = hidden_channels

        self.sinc_layer = SincConv(channel1, sinc_filter_length)
        self.bn1 = nn.BatchNorm1d(channel1)

        self.ResBlocks = nn.Sequential(
            ResBlock(channel1, channel1),
            ResBlock(channel1, channel1),
            ResBlock(channel1, channel2),
            ResBlock(channel2, channel2),
            ResBlock(channel2, channel2),
            ResBlock(channel2, channel2)
        )

        self.bn2 = nn.BatchNorm1d(channel2)
        self.gru = nn.GRU(channel2, gru_dim, num_layers=3, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(gru_dim, gru_dim),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Linear(gru_dim, 2)
        )

    def forward(self, audio, **batch):
        # x: (B, T)
        audio = audio.unsqueeze(1) # (B, 1, T)
        x = self.sinc_layer(audio) # (B, F, T)
        x = F.max_pool1d(torch.abs(x), 3)
        x = F.leaky_relu(self.bn1(x), NEGATIVE_SLOPE)

        x = self.ResBlocks(x)
        x = F.leaky_relu(self.bn2(x), NEGATIVE_SLOPE) # (B, F, T)
        x = x.transpose(1, 2) # (B, T, F)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.head(x) # 
        
        return {"logits": x}
