import torch
from torchvision import models
from torchviz import make_dot

# 构建ResNet50模型
model = models.resnet50(pretrained=True)

# 生成一个虚拟输入
x = torch.randn(1, 3, 224, 224)

# 前向传播
y = model(x)

# 绘制模型架构图a
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('resnet50_architecture')
