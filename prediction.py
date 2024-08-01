import os
from PIL import Image
import torch
from torchvision import models, transforms
from tqdm import tqdm

# 定义测试数据集类
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith(('.tif', '.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# 加载最佳模型权重
def load_best_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # 3 classes: NC, Positive, Weakly positive
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 对测试集进行预测
def predict(model, test_loader, device):
    model = model.to(device)
    predictions = []
    class_names = ["NC", "Positive", "Weakly positive"]

    with torch.no_grad():
        for inputs, img_paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for img_path, pred in tqdm(zip(img_paths, preds)):
                predictions.append((os.path.basename(img_path), class_names[pred.item()]))

    return predictions

# 保存预测结果到文本文件
def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for img_name, pred in predictions:
            f.write(f"{img_name}: {pred}\n")

# 主程序
def main(test_dir, model_path, output_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(test_dir, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_best_model(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictions = predict(model, test_loader, device)
    save_predictions(predictions, output_file)


# 替换为实际的路径
# test_dir = 'datasets/test_nolabel/test_linchuang'
test_dir = 'datasets/test_nolabel/test_tidu'
model_path = 'weights/best_model_epoch_9_val_acc_1.0000.pth'
output_file = 'predictions/prediction.txt'

main(test_dir, model_path, output_file)
