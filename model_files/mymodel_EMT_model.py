import torch
import torch.nn as nn

from .fave_vtn_encoder import VTN
from .pose_MSG3D_encoder.msg3d import Model as PoseModel
from .mymodel_EMT import EMTModel
#from gaze_global_mapper import GazeGlobalMapper

class LatentQueryPool(nn.Module):
    """
    学习 L_out 个查询，做 cross-attn:  [B, L_in, D] -> [B, L_out, D]
    """
    def __init__(self, D: int, L_out: int, num_heads=4, dropout=0.1):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(L_out, D))
        self.attn   = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(D)

    def forward(self, x):  # [B, L_in, D]
        B, L_in, D = x.shape
        Q = self.latent.unsqueeze(0).expand(B, -1, -1)  # [B, L_out, D]
        y, _ = self.attn(Q, x, x)                       # [B, L_out, D]
        return self.norm(y)

class SeqAlignQuery(nn.Module):
    """
    把 [B, L_in, D_in] 映射到 [B, L_out, D_out]：线性 -> query pool -> 线性
    """
    def __init__(self, D_in, D_out, L_out, heads=4, dropout=0.1):
        super().__init__()
        self.pre   = nn.Linear(D_in, D_out) if D_in != D_out else nn.Identity()
        self.pool  = LatentQueryPool(D_out, L_out, num_heads=heads, dropout=dropout)
        self.post  = nn.Identity()
    def forward(self, x):
        x = self.pre(x)      # [B, L_in, D_out]
        x = self.pool(x)     # [B, L_out, D_out]
        return x

class ModalityProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.proj(x)


class MultiModalEMT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.frames = config.frames
        self.gaze_global_dim = config.gaze_global_dim       # Dg
        self.gaze_input_dim = config.gaze_input_dim         # D（投影后统一维度）

        """  # 可选：把每折训练集拟合的 mu/std 传进来做标准化
        self.gaze_global_mapper = GazeGlobalMapper(
            global_dim=self.gaze_global_dim,
            out_dim=self.gaze_input_dim,   # 先映射到你“gaze 输入维”(D)，再接 ModalityProjector
            hidden_dims=(256, 256),
            activation="gelu",
            dropout=0.1,
            normalize=getattr(config, "normalize_gaze_global", False),
            mu=getattr(config, "mu_g", None),
            std=getattr(config, "std_g", None),
            zero_out_timeseries_global_dims=True,
            replace_row=True
        ) """

        # === 面部视觉编码器（ViT + Transformer） ===
        self.face_encoder = VTN(
            frames=config.frames,
            num_classes=config.num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
            spatial_frozen=True,
            spatial_size=config.spatial_size,
            spatial_args=config.spatial_args,
            temporal_type=config.temporal_type,
            temporal_args=config.temporal_args,
        )

        # === 姿态编码器（MSG3D） ===
        self.pose_encoder = PoseModel(
            num_class=config.num_classes,
            num_point=25,
            num_person=1,
            num_gcn_scales=13,
            num_g3d_scales=6,
            graph=config.graph
        )

        # === 眼动输入维度（默认 8） ===
        gaze_input_dim = config.gaze_input_dim

        """ # === 投影器，将所有模态统一到 hidden_dim 维度 ===
        self.face_proj = ModalityProjector(input_dim=768, output_dim=self.hidden_dim)
        self.pose_proj = ModalityProjector(input_dim=384, output_dim=self.hidden_dim)
        self.gaze_proj = ModalityProjector(input_dim=gaze_input_dim, output_dim=self.hidden_dim) """

        L_out = config.frames

        # 假设统一到 L_out = 60，统一维度到 hidden_dim = D
        self.face_align = SeqAlignQuery(D_in=768, D_out=self.hidden_dim, L_out=L_out)
        self.pose_align = SeqAlignQuery(D_in=384, D_out=self.hidden_dim, L_out=L_out)
        self.gaze_align = SeqAlignQuery(D_in=config.gaze_input_dim, D_out=self.hidden_dim, L_out=L_out)

        # === TensorFormer 融合模块 ===
        self.EMT = EMTModel(
            hidden_dim=self.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            num_classes=config.num_classes,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

    def forward(self, face_imgs, pose, gaze):
        """
        face_imgs: [B, F, C, H, W]
        pose:      [B, 3, T, V, 1]
        gaze:      [B, T, gaze_dim]
        """
        # gaze: [B, T, D_in]，第0行为全局token
        #gaze, g_proj = self.gaze_global_mapper(gaze)  # 回写后的 gaze 仍是 [B, T, D_in或out_dim]

        # --- 编码阶段 ---
        face_feat = self.face_encoder.forward_new(face_imgs)  # [B, F, 768]
        pose_feat = self.pose_encoder.forward_new(pose)       # [B, F, 384]
        #print(pose_feat.shape)
        #print(face_feat.shape)

        """ # --- 统一维度 ---
        face_feat = self.face_proj(face_feat)   # [B, T, D]
        pose_feat = self.pose_proj(pose_feat)   # [B, T, D]
        gaze_feat = self.gaze_proj(gaze)        # [B, T, D] """

        # --- 统一维度 ---
        face_feat = self.face_align(face_feat)   # [B, T, D]
        pose_feat = self.pose_align(pose_feat)   # [B, T, D]
        gaze_feat = self.gaze_align(gaze)        # [B, T, D]

        # --- TensorFormer 融合 ---
        logits = self.EMT(face_feat, pose_feat, gaze_feat)

        return logits
"""
if __name__ == "__main__":

    import torch

    # ==== 指定设备 ====
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # ==== 模拟配置 ====
    class DummyConfig:
        hidden_dim = 128
        frames = 60
        num_classes = 2
        img_size = 224
        patch_size = 16
        spatial_frozen = True
        spatial_size = 'base'
        spatial_args = {}
        temporal_type = 'longformer'
        temporal_args = {
            'seq_len': 60,
            'dim': 768,
            'depth': 3,
            'heads': 12,
            'dim_head': 64,
            'mlp_dim': 3072,
            "attention_window": 8,
            'dropout': 0.1
        }
        gaze_input_dim = 8
        ffn_hidden_dim = 256
        num_layers = 4
        tau = 1.0
        dropout = 0.1
        graph = 'graph.kinetics.AdjMatrixGraph'

    config = DummyConfig()

    # ==== 初始化模型 ====
    model = MultiModalTensorFormer(config).to(device)
    model.eval()

    # ==== 构造模拟输入 ====
    B, F, C, H, W =8, config.frames, 3, config.img_size, config.img_size
    V, M = 25, 1
    gaze_dim = config.gaze_input_dim

    dummy_face = torch.randn(B, F, C, H, W).to(device)
    dummy_pose = torch.randn(B, 3, F, V, M).to(device)
    dummy_gaze = torch.randn(B, 120, gaze_dim).to(device)

    # ==== 前向传播 ====
    with torch.no_grad():
        output = model(dummy_face, dummy_pose, dummy_gaze)

    print("✅ 模型输出 logits 维度:", output.shape)  # 应为 [B, num_classes]
"""