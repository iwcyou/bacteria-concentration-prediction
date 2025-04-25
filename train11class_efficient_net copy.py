"""
Author: Kun Feng
Date: 2025/3/19
Description: 病毒浓度分类模型训练（11个有序类别）
改进版：增强颜色敏感性的模型和评估指标
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from efficientnet_pytorch import EfficientNet  # pip install efficientnet-pytorch
import random

# ----------------------
# 2. 定义分类标签
# ----------------------
class_names = [
    "more_than_5ug", "5ug", "0.5ug", "0.05ug",
    "5ng", "0.5ng", "0.05ng", "5pg",
    "0.5pg", "0.05pg_0.5fg", "Negative"
]
label_map = {cls: idx for idx, cls in enumerate(class_names)}

# ----------------------
# 3. 自定义数据集
# ----------------------
class CustomDataset(Dataset):
    """
    Expects folder structure:

    root_dir/
      train/         <-- split name
         classA/
           img1.jpg
           img2.png
         classB/
           ...
      test/          <-- split name
         classA/
         classB/
      (optional) val/
    """
    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: path to the parent (e.g. "datasets/class_11")
        split: one of the subfolders, e.g. "train", "val", "test"
        transform: torchvision transforms to apply
        """
        self.samples = []
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Split folder not found: {split_dir}")

        for cls_name in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            if cls_name not in label_map:
                # warn if there's a mis‑named folder
                print(f"Warning: '{cls_name}' not in label_map, skipping")
                continue
            label = label_map[cls_name]
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    self.samples.append((os.path.join(cls_path, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}! Check folder names.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ----------------------
# 4. 改进的数据预处理（增强颜色增强）
# ----------------------
def preprocess_data(root_dir, batch_size=32, val_split=0.2, seed=42):
    """
    root_dir/
      train/
        classA/
        classB/
        ...
      test/
        classA/
        classB/
        ...
    val_split: 从 train 中抽取验证集比例
    返回: train_loader, val_loader, test_loader
    """
    # 1. 定义 transform
    color_transform = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    )
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomApply([color_transform], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, shear=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # 2. 加载完整的 train 数据集（用于再划分）
    train_full = CustomDataset(root_dir, split="train", transform=transform)
    total = len(train_full)
    if total == 0:
        raise RuntimeError(f"No images found in {os.path.join(root_dir,'train')}")

    # 3. 根据 val_split 计算 train/val 大小
    val_size = int(total * val_split)
    train_size = total - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError("val_split 导致 train_size 或 val_size 为 0，请调整 val_split 大小。")

    # 4. 划分
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=generator)

    # 5. 加载 test 数据集
    test_ds = CustomDataset(root_dir, split="test", transform=transform)
    if len(test_ds) == 0:
        raise RuntimeError(f"No images found in {os.path.join(root_dir,'test')}")

    # 6. DataLoader
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ----------------------
# 5. 构建颜色敏感的EfficientNet模型
# ----------------------
def build_model():
    model = EfficientNet.from_pretrained('efficientnet-b3')
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.label_map = label_map
    return model

# ----------------------
# 6. 改进的训练过程（添加评估指标）
# ----------------------
def train_model(model, train_loader, val_loader, epochs=50, learning_rate=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs+1):
        # --- Training ---
        model.train()
        losses, preds, targets = [], [], []
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(labs.cpu().numpy())

        train_metrics = calculate_metrics(targets, preds)
        avg_train_loss = np.mean(losses)

        # --- Validation ---
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_metrics = calculate_metrics(val_labels, val_preds)
        scheduler.step(val_metrics['f1'])
        print(f"Epoch {epoch}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Train F1: {train_metrics['f1']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")

        # log
        metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k,v in train_metrics.items()},
            **{f'val_{k}': v for k,v in val_metrics.items()}
        }
        history.append(metrics)
        wandb.log(metrics)


        # save best
        os.makedirs('weights', exist_ok=True)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            fname = f"weights/best_model_epoch{epoch}_f1{best_f1:.4f}.pth"
            torch.save(model.state_dict(), fname)
            print(f"New best model saved: {fname}")

    return model, history

def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            losses.append(criterion(out, labs).item())
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(labs.cpu().numpy())
    return np.mean(losses), preds, labels

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# ----------------------
# 7. 改进的评估函数（添加详细分类报告）
# ----------------------
def evaluate_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    metrics = calculate_metrics(labels, preds)

    # classification report
    report_dict = classification_report(labels, preds, target_names=class_names, digits=4, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print("\nClassification Report:\n", report_df)

    # log to wandb
    wandb.log({"test_metrics": metrics})
    wandb.log({"classification_report": wandb.Table(dataframe=report_df)})

    # confusion matrix
    plot_confusion_matrix(labels, preds)

    return metrics

def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

# ----------------------
# 8. 主程序
# ----------------------
if __name__ == "__main__":

    # ----------------------
    # 1. 配置 wandb
    # ----------------------
    wandb.init(project="virus-classification-enhanced")

    root_dir = "datasets/class_11"
    train_loader, val_loader, test_loader = preprocess_data(root_dir, batch_size=16, val_split=0.2, seed=42)
    model = build_model()

    config = {
        "epochs": 100,
        "learning_rate": 2e-4,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "model": "EfficientNet-B3",
        "augmentation": "Enhanced ColorJitter"
    }
    wandb.config.update(config)

    trained_model, history = train_model(
        model, train_loader, val_loader,
        epochs=config["epochs"], learning_rate=config["learning_rate"]
    )

    print("\nFinal Evaluation on Test Set:")
    test_metrics = evaluate_model(trained_model, test_loader)
    wandb.finish()
