import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

# 1. 定义与训练时相同的类别标签
class_names = [
    "more_than_5ug",
    "5ug",
    "0.5ug",
    "0.05ug",
    "5ng",
    "0.5ng",
    "0.05ng",
    "5pg",
    "0.5pg",
    "0.05pg_0.5fg",
    "Negative"
]

# 2. 定义与训练时相同的预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 加载自定义数据集类（与训练时相同）
class CustomDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        label_map = {class_name: i for i, class_name in enumerate(class_names)}

        for label_name, label_idx in label_map.items():
            folder_path = os.path.join(root_dir, label_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg')):
                        self.img_paths.append(img_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# 4. 构建模型函数（与训练时相同）
def build_model():
    model = models.resnet50(weights=None)  # 不加载预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    return model

# 5. 加载模型参数
def load_model(model_path):
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 6. 测试函数
def test_model(model, test_loader, save_path='confusion_matrix_test.png'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 绘制并保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_df.columns, yticklabels=cm_df.index)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.savefig(save_path)
    plt.close()

    return accuracy, cm_df

# 7. 主函数
def main():
    # 配置参数
    test_data_dir = 'datasets/class_11/test'  # 测试集路径
    model_path = 'weights/best_model_epoch_48_val_acc_0.8824.pth'     # 训练好的模型路径

    # 创建测试数据集
    test_dataset = CustomDataset(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # 测试模型
    accuracy, cm_df = test_model(model, test_loader)

    # 保存预测结果
    cm_df.to_csv('confusion_matrix_results.csv')
    print("Confusion matrix saved to confusion_matrix_results.csv")
    print("Confusion matrix visualization saved to confusion_matrix_test.png")

if __name__ == "__main__":
    main()
