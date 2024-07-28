import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import re

# 数据变换
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 加载训练好的模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load("./model/vgg16-397923af.pth"))

# 修改最后的全连接层
vgg.classifier[6] = nn.Linear(4096, 3)
vgg.load_state_dict(torch.load("./train_result/exp_2/vgg16_color_model.pth"))


# 预测单张图片的RGB值
def predict_color(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    vgg.eval()
    with torch.no_grad():
        output = vgg(image)
    return output.squeeze().numpy()  # 返回预测的RGB值


# 预测文件夹下所有图片的RGB值
def predict_colors_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predicted_rgb = predict_color(image_path)
            predicted_rgb = np.clip(np.round(predicted_rgb), 0, 255).astype(
                int
            )  # 限定在[0,255]并四舍五入

            # 打印文件名和预测结果
            print(f"Filename: {filename}")
            print(f"Predicted RGB values: {predicted_rgb}")


# 示例预测文件夹下的所有图片
image_folder_path = "image"  # 修改为你的图片文件夹路径
predict_colors_in_folder(image_folder_path)
