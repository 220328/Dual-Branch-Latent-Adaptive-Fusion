import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from torch.cuda.amp import autocast, GradScaler

from autism_dataset_v2 import AutismMultiModalDataset
from model_files.mymodel import MultiModalTensorFormer


# ===== 1. 配置 =====
class Config:
    hidden_dim = 64
    frames = 60
    num_classes = 2
    img_size = 224
    patch_size = 16

    spatial_size = 'base'
    spatial_args = {"pretrain_path": r"/Tri-Model_Fusion/jx_vit_base_p16_224-80ecf9dd.pth.bin"}
    temporal_type = 'longformer'
    temporal_args = {
        'seq_len': 60,
        'dim': 768,
        'depth': 2,
        'heads': 8,
        'dim_head': 64,
        'mlp_dim': 1024,
        'attention_window': 8,
        'dropout': 0.5
    }

    gaze_input_dim = 9
    ffn_hidden_dim = 128
    num_layers = 1
    tau = 1.0
    dropout = 0.45
    graph = 'model_files.graph.kinetics.AdjMatrixGraph'


# ===== 2. 设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)


# ===== 3. 数据路径 =====
root_dir = r"/Tri-Model_Fusion/data/data"
fold_id = 0

label_file  = os.path.join(root_dir, "task1_all_data.txt")
train_split = os.path.join(root_dir, "fold_split", f"fold{fold_id}", "train.txt")
val_split   = os.path.join(root_dir, "fold_split", f"fold{fold_id}", "val.txt")
test_split  = os.path.join(root_dir, "fold_split", f"fold{fold_id}", "test.txt")


# ===== 4. 图像预处理 =====
transform = transforms.Compose([
    transforms.ToTensor()
])


# ===== 5. 数据集 =====
train_dataset = AutismMultiModalDataset(
    root_dir=root_dir, split_file=train_split,
    label_file=label_file, transform=transform
)
val_dataset = AutismMultiModalDataset(
    root_dir=root_dir, split_file=val_split,
    label_file=label_file, transform=transform
)
test_dataset = AutismMultiModalDataset(
    root_dir=root_dir, split_file=test_split,
    label_file=label_file, transform=transform
)


# ===== 6. DataLoader =====
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=2, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=2, shuffle=False, num_workers=0)


# ===== 7. 模型与优化器 =====
config = Config()
model  = MultiModalTensorFormer(config).to(device)

class_weights = torch.tensor([1.2, 1.0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

num_epochs = 15
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

scaler = torch.amp.GradScaler('cuda')


# ===== 8. 训练配置 =====
best_val_acc = 0.0
save_path = os.path.join(os.path.dirname(__file__), "best_tensorformer.pth")

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []


# ===== 9. 单轮训练 =====
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Training", leave=False):
        face_imgs = batch["face_imgs"].to(device)
        pose      = batch["pose"].to(device)
        gaze      = batch["gaze"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(face_imgs, pose, gaze)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        del face_imgs, pose, gaze, labels, logits, loss

    return total_loss / total, correct / total


# ===== 10. 验证 =====
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            face_imgs = batch["face_imgs"].to(device)
            pose      = batch["pose"].to(device)
            gaze      = batch["gaze"].to(device)
            labels    = batch["label"].to(device)

            with autocast():
                logits = model(face_imgs, pose, gaze)
                loss   = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            del face_imgs, pose, gaze, labels, logits, loss

    return total_loss / total, correct / total


# ===== 11. 测试 =====
def test_model(model, loader, device):
    model.eval()
    all_sample_ids, all_true, all_pred, all_prob_0, all_prob_1 = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            face_imgs = batch["face_imgs"].to(device)
            pose      = batch["pose"].to(device)
            gaze      = batch["gaze"].to(device)
            labels    = batch["label"].to(device)

            with autocast():
                logits = model(face_imgs, pose, gaze)
                probs  = torch.softmax(logits, dim=1)
                preds  = torch.argmax(probs, dim=1)

            all_sample_ids.extend(batch["sample_id"])
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_prob_0.extend(probs[:, 0].cpu().numpy().tolist())
            all_prob_1.extend(probs[:, 1].cpu().numpy().tolist())

            del face_imgs, pose, gaze, labels, logits, probs, preds

    return all_sample_ids, all_true, all_pred, all_prob_0, all_prob_1


# ===== 12. 训练循环 =====
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] (LR: {current_lr:.7f}) "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"保存最佳模型到: {save_path}")

    gc.collect()
    torch.cuda.empty_cache()

print("\n训练结束，最佳验证集准确率:", best_val_acc)


# ===== 13. 测试 =====
print("\n开始加载最佳模型进行测试...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

sample_ids, y_true, y_pred, prob_0, prob_1 = test_model(model, test_loader, device)
print("测试完成，样本数:", len(sample_ids))


# ===== 14. 测试指标 =====
acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, prob_1)
except ValueError:
    auc = None

cm = confusion_matrix(y_true, y_pred)

print("\n===== Test Metrics =====")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test F1-score : {f1:.4f}")
if auc is not None:
    print(f"Test AUC      : {auc:.4f}")
else:
    print("Test AUC      : 无法计算（测试集可能只有单一类别）")
print("Confusion Matrix:")
print(cm)


# ===== 15. 导出结果 =====
result_df = pd.DataFrame({
    "sample_id": sample_ids,
    "true_label": y_true,
    "pred_label": y_pred,
    "prob_0": prob_0,
    "prob_1": prob_1
})
result_csv_path = os.path.join(os.path.dirname(__file__), "test_results.csv")
result_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
print(f"\n测试结果已保存到: {result_csv_path}")


# ===== 16. 训练曲线 =====
epochs = range(1, num_epochs + 1)

loss_curve_path = os.path.join(os.path.dirname(__file__), "loss_curve.png")
acc_curve_path  = os.path.join(os.path.dirname(__file__), "acc_curve.png")

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses,   label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs,   label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig(acc_curve_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Loss 曲线已保存到: {loss_curve_path}")
print(f"Accuracy 曲线已保存到: {acc_curve_path}")
