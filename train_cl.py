import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


# 初始化wandb
wandb.init(project="bacteria-classification")

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.label_map = {"NC": 0, "Positive": 1, "Weakly_positive": 2}

        for label_name, label in self.label_map.items():
            folder_path = os.path.join(root_dir, label_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path.endswith(('.tif', '.jpg', '.png')):
                        self.img_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def preprocess_data(root_dir, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(os.path.join(root_dir, 'train'), transform=transform)
    test_dataset = CustomDataset(os.path.join(root_dir, 'test'), transform=transform)

    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 构建模型
def build_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: NC, Positive, Weakly positive
    model.label_map = {"NC": 0, "Positive": 1, "Weakly_positive": 2}  # Save the label map
    return model


# 训练模型
def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_acc = 0.0
    best_epoch = 0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "train_acc": epoch_acc, "val_loss": val_loss, "val_acc": val_acc})

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, f"weights/best_model_epoch_{best_epoch}_val_acc_{best_acc:.4f}.pth")

    model.load_state_dict(best_model_wts)
    return model

# 评估模型
def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_corrects = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(preds.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    test_acc_score = accuracy_score(all_labels, all_outputs)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Accuracy Score: {test_acc_score:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_outputs)
    cm_df = pd.DataFrame(cm, index=[i for i in model.label_map.keys()], columns=[i for i in model.label_map.keys()])

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    # plt.show()


# 主程序
root_dir = 'datasets/dataset3class'  # 替换为数据集所在目录
train_loader, val_loader, test_loader = preprocess_data(root_dir)
model = build_model()
model = train_model(model, train_loader, val_loader)
evaluate_model(model, test_loader)

wandb.finish()
