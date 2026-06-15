import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPoolingNet(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x: [B, T, D], mask: [B, T]
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask
            sum_x = x.sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-5)
            pooled = sum_x / count
        else:
            pooled = x.mean(dim=1)
            
        return self.classifier(pooled)

class StatisticalPoolingNet(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, mask=None):
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) # [B, T, 1]
            x_masked = x * mask_expanded
            
            # Mean
            sum_x = x_masked.sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-5)
            mean = sum_x / count
            
            # Variance / Std
            var = ((x_masked - mean.unsqueeze(1)) ** 2 * mask_expanded).sum(dim=1) / count
            std = torch.sqrt(var + 1e-6)
            
            # Max and Min (need to handle padding carefully for max/min)
            # Replace padded values with very small/large numbers
            x_max_masked = x.masked_fill(mask_expanded == 0, -1e9)
            max_val, _ = x_max_masked.max(dim=1)
            
            x_min_masked = x.masked_fill(mask_expanded == 0, 1e9)
            min_val, _ = x_min_masked.min(dim=1)
            
            pooled = torch.cat([mean, std, max_val, min_val], dim=1)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            max_val, _ = x.max(dim=1)
            min_val, _ = x.min(dim=1)
            pooled = torch.cat([mean, std, max_val, min_val], dim=1)
            
        return self.classifier(pooled)

class SelfAttentionPoolingNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, mask=None):
        attn_logits = self.attention(x) # [B, T, 1]
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return self.classifier(pooled)

class TransformerPoolingNet(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64, num_heads=4, num_layers=1, num_classes=2):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=bottleneck_dim, nhead=num_heads, batch_first=True, dropout=0.5)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token approach
        self.cls_token = nn.Parameter(torch.randn(1, 1, bottleneck_dim))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x, mask=None):
        # Reduce dimensionality to prevent massive overfitting
        x = self.bottleneck(x)
        
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if mask is not None:
            # mask: [B, T] (1 is valid, 0 is pad)
            padding_mask = (mask == 0)
            # Prepend False for the CLS token
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        else:
            padding_mask = None
            
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        # Take CLS token output
        pooled = out[:, 0, :]
        return self.classifier(pooled)

class BiGRUPoolingNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, mask=None):
        out, _ = self.gru(x)
        
        attn_logits = self.attention(out)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(out * attn_weights, dim=1)
        return self.classifier(pooled)

class NetVLADPoolingNet(nn.Module):
    def __init__(self, input_dim=768, bottleneck_dim=64, num_clusters=2, num_classes=2):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = bottleneck_dim
        
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.conv = nn.Conv1d(bottleneck_dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, bottleneck_dim))
        
        # Intra-normalization, L2 normalization yields output dim = num_clusters * bottleneck_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_clusters * bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x: [B, T, D]
        x = self.bottleneck(x)
        B, T, D = x.size()
        
        # soft-assignment
        # [B, num_clusters, T]
        x_trans = x.transpose(1, 2)
        soft_assign = self.conv(x_trans)
        
        if mask is not None:
            # mask: [B, T] -> [B, 1, T]
            mask_exp = mask.unsqueeze(1)
            soft_assign = soft_assign.masked_fill(mask_exp == 0, -1e9)
            
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # residuals
        vlad = torch.zeros([B, self.num_clusters, D], dtype=x.dtype, device=x.device)
        for C in range(self.num_clusters):
            # residual: [B, T, D]
            residual = x - self.centroids[C:C+1, :].expand(x.size(0), x.size(1), -1)
            # weight: [B, T, 1]
            weight = soft_assign[:, C:C+1, :].transpose(1, 2)
            if mask is not None:
                residual = residual * mask.unsqueeze(-1)
            vlad[:, C:C+1, :] = torch.sum(residual * weight, dim=1, keepdim=True)
            
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(B, -1)               # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        
        return self.classifier(vlad)
