import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------
# 定义五分类标签
# ----------------------
class_names = [
    "5ng_5pg",
    "5pg_0.5fg",
    "5ug_5ng",
    "more_than_5ug",
    "Negative"
]
label_map = {class_name: i for i, class_name in enumerate(class_names)}

# ----------------------
# 自定义数据集类
# ----------------------
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for label_name, label_idx in label_map.items():
            folder_path = os.path.join(root_dir, label_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path.endswith(('.tif', '.jpg', '.png', '.jpeg')):
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

# ----------------------
# 预处理与加载数据
# ----------------------
def load_test_data(test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = CustomDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# ----------------------
# 构建并加载模型
# ----------------------
def load_model(weights_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model

# ----------------------
# 模型评估与混淆矩阵
# ----------------------
def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}, Loss: {total_loss / len(test_loader.dataset):.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

# ----------------------
# 主函数
# ----------------------
if __name__ == "__main__":
    test_dir = "datasets/H1N1/test"  # 修改为你的测试集路径
    weights_path = "weights/best_model_epoch_27_val_acc_1.0000.pth"  # 替换为你的模型权重文件路径

    test_loader = load_test_data(test_dir)
    model = load_model(weights_path)
    evaluate(model, test_loader)
