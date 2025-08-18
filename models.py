import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (B, T, D)
        weights = self.attn(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(weights, dim=1)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)
        return out


class ResNet_BiGRU_Net(nn.Module):
    def __init__(self, meta_features=4, handcrafted_dim=4):
        super(ResNet_BiGRU_Net, self).__init__()
        self.in_channels = 64

        self.initial_conv = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.bi_gru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        self.attn_pool = AttentionPooling(256)

        self.aux_proj  = nn.Linear(meta_features + handcrafted_dim, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.norm_q = nn.LayerNorm(256)
        self.norm_kv = nn.LayerNorm(256)

        self.pool = nn.AdaptiveAvgPool1d(1)  # pooled residual
        self.dropout = nn.Dropout(0.5)

        self.mlp  = nn.Sequential(
            nn.Linear(512, 50),  # cross-attn out + pooled main
            nn.GELU(),
            nn.Linear(50, 1)  # binary logit 
        )
#        self.fc = nn.Linear(256, 1)  # 메타 피처가 attention으로 통합되므로 차원 변경

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = [BasicBlock1D(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, signals, ages, sexes, handcrafted):
        x = self.initial_relu(self.initial_bn(self.initial_conv(signals)))
        x = self.layer1(x) # (B, D, T)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # (B, 256, 512)

        x = x.permute(0, 2, 1)  # (B, T, D)
        x, _ = self.bi_gru(x)   # (B, T, D) (B, 256, 512)
        B, T, _ = x.shape # (B,256, 256) (B, T, hiddenDim*2)

        # K,V from main sequence
        K = self.norm_kv(x)  # (B, T, 256)
        V = K

        # Q from aux vector (single token)
        aux_features = self.aux_proj(torch.cat([ages, sexes, handcrafted], dim=1))
        Q = self.norm_q(aux_features).unsqueeze(1)  # (B, 1, 256)

        # Cross-Attention: Q attends over main sequence
        out, _ = self.attn(Q, K, V, need_weights=False)  # (B, 1, 256)
        out = out.squeeze(1)  # (B, 256)

        # x = self.attn_pool(x)   # (B, 256)
 
        # Residual pooled main
        pooled_main = self.pool(K.transpose(1, 2)).squeeze(-1)  # (B, 256)

        # Fusion
        fused = torch.cat([out, pooled_main], dim=-1)  # (B, 512)
        logits = self.mlp(fused)  # (B, 1)

        return logits
    