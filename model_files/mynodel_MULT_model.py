import torch
import torch.nn as nn

from .fave_vtn_encoder import VTN
from .pose_MSG3D_encoder.msg3d import Model as PoseModel
from .mymodel_MULT import MULTModel
from types import SimpleNamespace

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


class MultiModalMULT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.frames = config.frames

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

        # === 投影器，将所有模态统一到 hidden_dim 维度 ===
        self.face_proj = ModalityProjector(input_dim=768, output_dim=self.hidden_dim)
        self.pose_proj = ModalityProjector(input_dim=384, output_dim=self.hidden_dim)
        self.gaze_proj = ModalityProjector(input_dim=gaze_input_dim, output_dim=self.hidden_dim)

        # === 构造 hyp_params 供 MULTModel 使用 ===
        hyp_params = SimpleNamespace(
            orig_d_l=self.hidden_dim,
            orig_d_a=self.hidden_dim,
            orig_d_v=self.hidden_dim,
            output_dim=config.num_classes,
            layers=config.num_layers,
            num_heads=8,                   # 可调
            attn_dropout=config.dropout,
            attn_dropout_a=config.dropout,
            attn_dropout_v=config.dropout,
            relu_dropout=config.dropout,
            res_dropout=config.dropout,
            out_dropout=config.dropout,
            embed_dropout=config.dropout,
            attn_mask=False,
            lonly=True,
            aonly=True,
            vonly=True
        )

        # === 融合模块（MulT） ===
        self.MULT = MULTModel(hyp_params)

    def forward(self, face_imgs, pose, gaze):
        """
        face_imgs: [B, F, C, H, W]
        pose:      [B, 3, T, V, 1]
        gaze:      [B, T, gaze_dim]
        """
        # --- 编码阶段 ---
        face_feat = self.face_encoder.forward_new(face_imgs)  # [B, F, 768]
        pose_feat = self.pose_encoder.forward_new(pose)       # [B, F, 384]

        # --- 统一维度 ---
        face_feat = self.face_proj(face_feat)   # [B, T, D]
        pose_feat = self.pose_proj(pose_feat)   # [B, T, D]
        gaze_feat = self.gaze_proj(gaze)        # [B, T, D]

        # --- TensorFormer 融合 ---
        logits, _ = self.MULT(face_feat, pose_feat, gaze_feat)

        return logits