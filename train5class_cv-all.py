"""
Author: Kun Feng (Modified for Full-Data 5-Fold CV)
Date: 2025/3/19
Description:
1. Merges 'train' and 'test' folders into one dataset.
2. Performs Stratified 5-Fold Cross Validation.
3. Generates an aggregated Confusion Matrix for all samples.
"""

import os
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

# ----------------------
# 1. 全局配置
# ----------------------
PROJECT_NAME = "virus-classification-full-cv"
ROOT_DIR = 'datasets/H1N1'  # 数据集根目录，下面应该有 'train' 和 'test'
BATCH_SIZE = 16             # 小样本建议 Batch Size 设小一点
N_FOLDS = 5                 # 5折
EPOCHS = 30                 # 每一折训练多少轮
LR = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = [
    "5ng_5pg",
    "5pg_0.5fg",
    "5ug_5ng",
    "more_than_5ug",
    "Negative"
]
label_map = {class_name: i for i, class_name in enumerate(class_names)}

# ----------------------
# 2. 合并数据集类 (核心修改)
# ----------------------
class CombinedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        这个类会自动遍历 root_dir 下的 'train' 和 'test' 两个文件夹，
        将所有图片合并到一个列表中。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # 我们要遍历的子目录列表
        sub_dirs = ['train', 'test']

        for sub_dir in sub_dirs:
            # 比如 datasets/H1N1/train
            base_path = os.path.join(root_dir, sub_dir)
            if not os.path.exists(base_path):
                print(f"Warning: {base_path} not found, skipping.")
                continue

            # 遍历 5 个类别的文件夹
            for label_name, label_idx in label_map.items():
                folder_path = os.path.join(base_path, label_name)
                if os.path.isdir(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)
                        if img_path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg')):
                            self.img_paths.append(img_path)
                            self.labels.append(label_idx)

        print(f"Total images loaded: {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑图防止崩溃，或者你可以做其他处理
            image = Image.new('RGB', (224, 224))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class PathLabelDataset(Dataset):
    """A lightweight dataset over an existing (img_paths, labels) list.

    Use this to apply different transforms (e.g., train augmentation vs eval preprocessing)
    while keeping the *same indices* for K-Fold splits.
    """

    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ----------------------
# 3. 模型构建
# ----------------------
def build_model():
    # 依然使用 ResNet50
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    return model

# ----------------------
# 4. 单折训练函数
# ----------------------
def train_one_fold(fold_index, train_idx, val_idx, train_dataset, eval_dataset):
    print(f"\n>>> Starting Fold {fold_index+1}/{N_FOLDS}")

    # 根据索引划分数据
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 初始化模型
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 记录该折的最佳结果
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化 wandb (可选)
    # run = wandb.init(project=PROJECT_NAME, group="k-fold", name=f"fold_{fold_index+1}", reinit=True)

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels.data).sum().item()

        val_acc = val_corrects / len(val_subset)

        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Fold {fold_index+1} Best Accuracy: {best_acc:.4f}")

    # 加载最佳权重进行最后的预测，用于绘制混淆矩阵
    model.load_state_dict(best_model_wts)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # run.finish()
    return best_acc, all_labels, all_preds

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == "__main__":
    # 训练用增强；验证/测试用固定预处理（否则每个 epoch / 每次统计都在随机变化）
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # 颜色抖动：模拟不同光照条件 (亮度、对比度)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 加载所有数据
    full_dataset = CombinedDataset(ROOT_DIR, transform=None)
    train_dataset = PathLabelDataset(full_dataset.img_paths, full_dataset.labels, transform=train_transform)
    eval_dataset = PathLabelDataset(full_dataset.img_paths, full_dataset.labels, transform=eval_transform)

    # 获取所有标签用于分层
    # 注意：Subset不直接暴露labels，所以我们需要从原始数据集中按索引取，
    # 但为了 KFold，我们直接取 full_dataset.labels 即可
    all_labels = np.array(full_dataset.labels)

    # 2. 准备 K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []

    # 用于存储所有折的预测结果，最后画一个总的混淆矩阵
    total_y_true = []
    total_y_pred = []

    # 3. 循环训练
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        acc, val_labels, val_preds = train_one_fold(fold, train_idx, val_idx, train_dataset, eval_dataset)

        fold_accuracies.append(acc)
        total_y_true.extend(val_labels)
        total_y_pred.extend(val_preds)

    # 4. 输出最终统计结果
    print("\n" + "="*30)
    print("5-Fold Cross Validation Results:")
    print(f"Accuracies: {fold_accuracies}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print("="*30)

    # 5. 绘制并保存总混淆矩阵 (Aggregated Confusion Matrix)
    # 这就是你要放在论文里的图，专门用来分析 Negative 样本
    cm = confusion_matrix(total_y_true, total_y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Aggregated Confusion Matrix (5-Fold CV)\nMean Acc: {np.mean(fold_accuracies):.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('total_confusion_matrix.png', dpi=300)
    print("Confusion matrix saved as 'total_confusion_matrix.png'")

    # 6. 打印分类报告 (Precision, Recall, F1-score)
    print("\nClassification Report:")
    print(classification_report(total_y_true, total_y_pred, target_names=class_names))
