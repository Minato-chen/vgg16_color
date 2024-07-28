import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import json


# 自定义数据集
class ColorDataset(Dataset):
    def __init__(self, image_paths, rgb_values, transform=None):
        self.image_paths = image_paths
        self.rgb_values = rgb_values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        rgb = self.rgb_values[idx]
        return image, torch.tensor(rgb, dtype=torch.float32)


# 数据变换
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 加载测试集
test_image_paths = np.load("./dataset/test_image_paths.npy", allow_pickle=True)
test_rgb_values = np.load("./dataset/test_rgb_values.npy", allow_pickle=True)
print(f"Test Dataset Size: {len(test_image_paths)}")

test_dataset = ColorDataset(test_image_paths, test_rgb_values, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载训练好的模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load("./model/vgg16-397923af.pth"))

# 修改最后的全连接层
vgg.classifier[6] = nn.Linear(4096, 3)
vgg.load_state_dict(torch.load("./train_result/exp_2/vgg16_color_model.pth"))

# 定义损失函数
criterion = nn.MSELoss()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# 测试模型
vgg.eval()
test_loss = 0.0
all_outputs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # 将 outputs 限制在 [0, 255] 的整数范围内
        outputs = torch.clamp(torch.round(outputs), 0, 255).int()
        labels = torch.clamp(torch.round(labels), 0, 255).int()

        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        print(f"Batch Loss: {loss.item()}")

# 打印总测试损失
print(f"Test Loss: {test_loss / len(test_loader)}")

# 保存结果
all_outputs_list = [output.tolist() for output in all_outputs]
all_labels_list = [label.tolist() for label in all_labels]

# 保存结果为 JSON 文件
with open("./test_results/test_outputs.json", "w") as f:
    json.dump(all_outputs_list, f)

with open("./test_results/test_labels.json", "w") as f:
    json.dump(all_labels_list, f)
