import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import wandb
from sklearn.metrics import mean_absolute_error, r2_score

# 初始化wandb
wandb.init(project="bacteria-concentration-prediction")

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                label = float(folder_name)
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
        return image, torch.tensor(label, dtype=torch.float64)

def preprocess_data(root_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(root_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 构建模型
def build_model():
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(1024, 1, dtype=torch.float64)
    )
    return model

# 训练模型
def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).double()

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).double(), labels.to(device).unsqueeze(1).double()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_mae += mean_absolute_error(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()) * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mae = running_mae / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.15f}, MAE: {epoch_mae:.15f}")

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).double(), labels.to(device).unsqueeze(1).double()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_mae += mean_absolute_error(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()) * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.15f}, Validation MAE: {val_mae:.15f}")

        wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "mae": epoch_mae, "val_loss": val_loss, "val_mae": val_mae})

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, f"weights/best_model_epoch_{best_epoch}_val_loss_{best_val_loss:.13f}.pth")

    model.load_state_dict(best_model_wts)
    return model

# 测试模型
def evaluate_model(model, test_loader):
    criterion = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).double()
    model.eval()

    test_loss = 0.0
    test_mae = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).double(), labels.to(device).unsqueeze(1).double()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_mae += mean_absolute_error(labels.cpu().numpy(), outputs.cpu().numpy()) * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    test_r2 = r2_score(all_labels, all_outputs)
    print(f"Test Loss: {test_loss:.15f}, Test MAE: {test_mae:.15f}, Test R²: {test_r2:.15f}")

# 主程序
root_dir = 'dataset'  # 替换为图片所在目录
train_loader, val_loader, test_loader = preprocess_data(root_dir)
model = build_model()
model = train_model(model, train_loader, val_loader)
evaluate_model(model, test_loader)

wandb.finish()
