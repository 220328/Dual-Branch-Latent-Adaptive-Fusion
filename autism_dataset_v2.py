import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class AutismMultiModalDataset(Dataset):
    def __init__(self, root_dir, split_file, label_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_frames = 60

        self.face_dir = os.path.join(root_dir, "task1_tensorformer_processed_faces")
        self.pose_dir = os.path.join(root_dir, "task1_npy_out_pose_60_tensorformer")
        self.gaze_dir = os.path.join(root_dir, "task1_gaze_data_npy")

        # 读取标签映射
        self.label_map = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample_id, label = line.split()
                self.label_map[sample_id] = int(label)

        # 读取划分样本ID
        self.sample_ids = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                sid = parts[0]
                self.sample_ids.append(sid)

    def __len__(self):
        return len(self.sample_ids)

    def _align_sequence(self, seq):
        """
        统一时序长度到 target_frames=60
        seq 第0维为时序维度 [T, ...]
        少于60帧末尾补0；多于60帧截取中间连续60帧
        """
        L = seq.shape[0]
        if L == self.target_frames:
            return seq
        if L < self.target_frames:
            pad_shape = (self.target_frames - L,) + seq.shape[1:]
            pad = torch.zeros(pad_shape, dtype=seq.dtype, device=seq.device)
            return torch.cat([seq, pad], dim=0)
        else:
            start = (L - self.target_frames) // 2
            return seq[start:start + self.target_frames]

    def _load_face(self, sample_id):
        sample_face_dir = os.path.join(self.face_dir, sample_id)
        img_files = sorted([
            os.path.join(sample_face_dir, x)
            for x in os.listdir(sample_face_dir)
            if x.lower().endswith(".png")
        ])
        # 调试用，排查帧数异常时打开
        # print(f"[DEBUG] 样本{sample_id} 人脸图片总数：{len(img_files)}")

        imgs = []
        for img_path in img_files:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                else:
                    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            imgs.append(img)

        face_raw = torch.stack(imgs, dim=0)
        face = self._align_sequence(face_raw)
        return face

    def _load_pose(self, sample_id):
        pose_path = os.path.join(self.pose_dir, sample_id + ".npy")
        pose = np.load(pose_path).astype(np.float32)  # [C=3, T, V=25, M=1]
        pose_tensor = torch.from_numpy(pose)

        # 置换到时序优先格式做对齐 [T, C, V, M]
        pose_t_first = pose_tensor.permute(1, 0, 2, 3)
        pose_t_aligned = self._align_sequence(pose_t_first)
        # 还原MSG3D标准输入维度 [C, T, V, M]
        pose_aligned = pose_t_aligned.permute(1, 0, 2, 3)
        return pose_aligned

    def _load_gaze(self, sample_id):
        gaze_path = os.path.join(self.gaze_dir, sample_id + ".npy")
        gaze = np.load(gaze_path).astype(np.float32)
        gaze_tensor = torch.from_numpy(gaze)
        seq_len = gaze_tensor.shape[0]

        if seq_len == self.target_frames:
            gaze_aligned = gaze_tensor
        elif seq_len == self.target_frames * 2:
            # 120帧相邻两帧均值下采样
            gaze_aligned = gaze_tensor.view(self.target_frames, 2, -1).mean(dim=1)
        else:
            # 异常帧数线性插值对齐
            gaze_input = gaze_tensor.unsqueeze(0).transpose(1, 2)
            gaze_aligned = F.interpolate(
                gaze_input,
                size=self.target_frames,
                mode='linear',
                align_corners=False
            )
            gaze_aligned = gaze_aligned.transpose(1, 2).squeeze(0)
        return gaze_aligned

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        face = self._load_face(sample_id)
        pose = self._load_pose(sample_id)
        gaze = self._load_gaze(sample_id)
        label = self.label_map[sample_id]

        return {
            "sample_id": sample_id,
            "face_imgs": face,
            "pose": pose,
            "gaze": gaze,
            "label": torch.tensor(label, dtype=torch.long)
        }