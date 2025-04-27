import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests

# ----------------------
# 定义标签和映射
# ----------------------
class_names = [
    "5ng_5pg",
    "5pg_0.5fg",
    "5ug_5ng",
    "more_than_5ug",
    "Negative"
]
concentration_map = {
    "5ng_5pg": "5ng-5pg",
    "5pg_0.5fg": "5pg-0.5fg",
    "5ug_5ng": "5ug-5ng",
    "more_than_5ug": "超过 5ug",
    "Negative": "阴性"
}
label_map = {i: name for i, name in enumerate(class_names)}

# ----------------------
# 图像预处理
# ----------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # batch 维度

# ----------------------
# 模型加载
# ----------------------
def load_model(weights_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# ----------------------
# 构造 prompt
# ----------------------
def generate_prompt(predicted_label):
    return f"请根据图像中检测到的甲型流感病毒浓度为【{predicted_label}】的情况，为患者提供诊疗建议。请你返回纯文本格式的建议，避免使用 HTML 或 Markdown 格式。请确保建议内容简洁明了，便于患者理解。"

# ----------------------
# 请求 DeepSeek API
# ----------------------
def ask_deepseek(prompt, api_key):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"DeepSeek API 调用失败: {response.status_code}, {response.text}")

# ----------------------
# 主流程
# ----------------------
def predict_single_image(image_path, weights_path):
    # api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key = "sk-76c43391f6564f4d813b0592112ab92a"
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY 以使用 DeepSeek API。")

    image_tensor = preprocess_image(image_path)
    model = load_model(weights_path)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_label = label_map[pred.item()]

    concentration = concentration_map[predicted_label]
    print(f"✅ 模型预测浓度等级为：{concentration}")
    prompt = generate_prompt(concentration)
    print(f"📨 提交的 Prompt：\n{prompt}")

    reply = ask_deepseek(prompt, api_key)
    print(f"\n🧑‍⚕️ DeepSeek 建议：\n{reply}")
    return prompt,reply


# ----------------------
# 使用方法示例
# ----------------------
if __name__ == "__main__":
    image_path = "datasets/H1N1/test/5ng_5pg/2.jpg"  # 替换为图像路径
    weights_path = "weights/best_model_epoch_27_val_acc_1.0000.pth"  # 替换为模型路径
    predict_single_image(image_path, weights_path)
