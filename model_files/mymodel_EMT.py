import torch
import torch.nn as nn
import torch.nn.functional as F


class MPU(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x_local, x_global):
        x = x_local + self.cross_attn(self.norm1(x_local), self.norm1(x_global), self.norm1(x_global))[0]
        x = x + self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x))[0]
        x = x + self.ffn(self.norm3(x))
        return x


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x)
        return pooled  # [B, 1, D]



class EMTBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3, share_direction=True):
        super().__init__()
        self.share_direction = share_direction

        if share_direction:
            self.mpu_l = MPU(hidden_dim, dropout)
            self.mpu_a = MPU(hidden_dim, dropout)
            self.mpu_v = MPU(hidden_dim, dropout)
        else:
            # 模态 l/a/v 的两个方向各用一套 MPU
            self.mpu_l_lg = MPU(hidden_dim, dropout)  # H_l → G
            self.mpu_l_gl = MPU(hidden_dim, dropout)  # G → H_l

            self.mpu_a_ag = MPU(hidden_dim, dropout)
            self.mpu_a_ga = MPU(hidden_dim, dropout)

            self.mpu_v_vg = MPU(hidden_dim, dropout)
            self.mpu_v_gv = MPU(hidden_dim, dropout)

        self.pool_W = nn.Linear(hidden_dim, hidden_dim)
        self.pool_v = nn.Parameter(torch.randn(hidden_dim))
        self.pool_b = nn.Parameter(torch.zeros(1))

    def attention_pool(self, G_l, G_a, G_v):
        G_cat = torch.stack([G_l.mean(dim=1), G_a.mean(dim=1), G_v.mean(dim=1)], dim=1)  # [B, 3, d]
        scores = torch.tanh(self.pool_W(G_cat) + self.pool_b)  # [B, 3, d]
        scores = torch.matmul(scores, self.pool_v)  # [B, 3]
        attn_weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(G_cat * attn_weights.unsqueeze(-1), dim=1)  # [B, d]
        return torch.stack([pooled, pooled, pooled], dim=1)  # [B, 3, d]

    def forward(self, H_l, H_a, H_v, G):
        if self.share_direction:
            H_l = self.mpu_l(H_l, G)
            G_l = self.mpu_l(G, H_l)

            H_a = self.mpu_a(H_a, G)
            G_a = self.mpu_a(G, H_a)

            H_v = self.mpu_v(H_v, G)
            G_v = self.mpu_v(G, H_v)
        else:
            H_l = self.mpu_l_lg(H_l, G)
            G_l = self.mpu_l_gl(G, H_l)

            H_a = self.mpu_a_ag(H_a, G)
            G_a = self.mpu_a_ga(G, H_a)

            H_v = self.mpu_v_vg(H_v, G)
            G_v = self.mpu_v_gv(G, H_v)

        G = self.attention_pool(G_l, G_a, G_v)
        return H_l, H_a, H_v, G

class EMTModel(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            EMTBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 加入初始 G 的 AttentionPool
        self.init_pool_t = AttentionPool(hidden_dim)
        self.init_pool_a = AttentionPool(hidden_dim)
        self.init_pool_v = AttentionPool(hidden_dim)


        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_t, X_a, X_v):
        G = torch.cat([
            X_t.mean(dim=1, keepdim=True),
            X_a.mean(dim=1, keepdim=True),
            X_v.mean(dim=1, keepdim=True)
        ], dim=1) 

        """ 
        # ✅ 使用 attention pooling 初始化 G
        pool_t = self.init_pool_t(X_t)  # [B, 1, D]
        pool_a = self.init_pool_a(X_a)
        pool_v = self.init_pool_v(X_v)

        G = torch.cat([pool_t, pool_a, pool_v], dim=1)  # [B, 3, D]
        """

        for layer in self.layers:
            X_t, X_a, X_v, G = layer(X_t, X_a, X_v, G)

        pooled_t = X_t.mean(dim=1)
        pooled_a = X_a.mean(dim=1)
        pooled_v = X_v.mean(dim=1)
        pooled_g = G.mean(dim=1)

        fused = torch.cat([pooled_t, pooled_a, pooled_v, pooled_g], dim=-1)
        #logits = self.predictor(fused)
        h_base = self.base_head(fused)
        h_final = h_base
        logits  = self.classifier(h_final)
        return logits
    

    def forward_MMAF(self, X_t, X_a, X_v):
        G = torch.cat([
            X_t.mean(dim=1, keepdim=True),
            X_a.mean(dim=1, keepdim=True),
            X_v.mean(dim=1, keepdim=True)
        ], dim=1)

        """ # ✅ 使用 attention pooling 初始化 G
        pool_t = self.init_pool_t(X_t)  # [B, 1, D]
        pool_a = self.init_pool_a(X_a)
        pool_v = self.init_pool_v(X_v)

        G = torch.cat([pool_t, pool_a, pool_v], dim=1)  # [B, 3, D] """

        for layer in self.layers:
            X_t, X_a, X_v, G = layer(X_t, X_a, X_v, G)

        """ pooled_t = X_t.mean(dim=1)
        pooled_a = X_a.mean(dim=1)
        pooled_v = X_v.mean(dim=1)
        pooled_g = G.mean(dim=1)

        fused = torch.cat([pooled_t, pooled_a, pooled_v, pooled_g], dim=-1)
        logits = self.predictor(fused) """
        return X_t, X_a, X_v, G

"""
if __name__ == "__main__":
    # Test run
    B, Tt, Ta, Tv, d = 4, 20, 30, 25, 128
    X_t = torch.randn(B, Tt, d)
    X_a = torch.randn(B, Ta, d)
    X_v = torch.randn(B, Tv, d)

    model = EMTModel(
        hidden_dim=128,
        ffn_hidden_dim=256,
        num_classes=2,
        num_layers=2,
        dropout=0.1
    )

    out = model(X_t, X_a, X_v)
    print("Output shape:", out.shape)  # Expected: [4, 2]
"""