import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchLatentBlock(nn.Module):
    """
    潜变量瓶颈 + 双分支互补挖掘融合模块。
    输入三路模态特征，输出融合后的三路特征。
    """
    def __init__(self, hidden_dim, ffn_hidden_dim, num_latents=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents

        # 潜变量池
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)

        # 潜变量聚合三模态信息
        self.gather_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gather_norm = nn.LayerNorm(hidden_dim)

        # 潜变量内部自注意力
        self.latent_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.latent_norm = nn.LayerNorm(hidden_dim)

        # 广播：模态向潜变量取共识信息（Q=X_m, K/V=L_fused）
        self.broadcast_attn_t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.broadcast_attn_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.broadcast_attn_v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.broadcast_norm_t = nn.LayerNorm(hidden_dim)
        self.broadcast_norm_a = nn.LayerNorm(hidden_dim)
        self.broadcast_norm_v = nn.LayerNorm(hidden_dim)

        # 共识分支 FFN
        self.ffn_shared_t = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)
        self.ffn_shared_a = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)
        self.ffn_shared_v = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)

        # 互补分支 FFN
        self.ffn_comp_t = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)
        self.ffn_comp_a = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)
        self.ffn_comp_v = self._build_ffn(hidden_dim, ffn_hidden_dim, dropout)

        # 可学习融合权重，控制共识/互补分支比例
        self.alpha_t = nn.Parameter(torch.zeros(1))
        self.alpha_a = nn.Parameter(torch.zeros(1))
        self.alpha_v = nn.Parameter(torch.zeros(1))

        self.ln_out_t = nn.LayerNorm(hidden_dim)
        self.ln_out_a = nn.LayerNorm(hidden_dim)
        self.ln_out_v = nn.LayerNorm(hidden_dim)

    def _build_ffn(self, d, d_hid, dropout):
        return nn.Sequential(
            nn.Linear(d, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d),
            nn.Dropout(dropout)
        )

    def forward(self, X_t, X_a, X_v):
        B = X_t.size(0)

        # 阶段 1：潜变量聚合与提炼
        X_all = torch.cat([X_t, X_a, X_v], dim=1)
        L = self.latents.unsqueeze(0).expand(B, -1, -1)

        L_gather, _ = self.gather_attn(L, X_all, X_all)
        L = self.gather_norm(L + L_gather)

        L_self, _ = self.latent_self_attn(L, L, L)
        L_fused = self.latent_norm(L + L_self)

        # 阶段 2：广播，提取共识特征
        X_t_shared_raw, _ = self.broadcast_attn_t(X_t, L_fused, L_fused)
        X_a_shared_raw, _ = self.broadcast_attn_a(X_a, L_fused, L_fused)
        X_v_shared_raw, _ = self.broadcast_attn_v(X_v, L_fused, L_fused)

        X_t_shared = self.broadcast_norm_t(X_t_shared_raw)
        X_a_shared = self.broadcast_norm_a(X_a_shared_raw)
        X_v_shared = self.broadcast_norm_v(X_v_shared_raw)

        # 互补残差
        X_t_comp = X_t - X_t_shared
        X_a_comp = X_a - X_a_shared
        X_v_comp = X_v - X_v_shared

        # 双分支加权融合
        alpha_t = torch.sigmoid(self.alpha_t)
        alpha_a = torch.sigmoid(self.alpha_a)
        alpha_v = torch.sigmoid(self.alpha_v)

        H_t = X_t + alpha_t * self.ffn_shared_t(X_t_shared) + (1 - alpha_t) * self.ffn_comp_t(X_t_comp)
        H_a = X_a + alpha_a * self.ffn_shared_a(X_a_shared) + (1 - alpha_a) * self.ffn_comp_a(X_a_comp)
        H_v = X_v + alpha_v * self.ffn_shared_v(X_v_shared) + (1 - alpha_v) * self.ffn_comp_v(X_v_comp)

        X_t_out = self.ln_out_t(H_t)
        X_a_out = self.ln_out_a(H_a)
        X_v_out = self.ln_out_v(H_v)

        return X_t_out, X_a_out, X_v_out


class TensorFormerModel(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, num_classes, num_layers=2, tau=1.0, dropout=0.1):
        super().__init__()

        num_latents = 16

        self.layers = nn.ModuleList([
            DualBranchLatentBlock(
                hidden_dim, ffn_hidden_dim,
                num_latents=num_latents, num_heads=4, dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_t, X_a, X_v, mask_t=None, mask_a=None, mask_v=None):
        for block in self.layers:
            X_t, X_a, X_v = block(X_t, X_a, X_v)

        pooled_t = X_t.mean(dim=1)
        pooled_a = X_a.mean(dim=1)
        pooled_v = X_v.mean(dim=1)

        fused = torch.cat([pooled_t, pooled_a, pooled_v], dim=-1)
        h_base = self.base_head(fused)
        logits = self.classifier(h_base)

        return logits

    def forward_MMAF(self, X_t, X_a, X_v, mask_t=None, mask_a=None, mask_v=None):
        for block in self.layers:
            X_t, X_a, X_v = block(X_t, X_a, X_v)
        return X_t, X_a, X_v
