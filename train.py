import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm  # 进度条
import os
import json
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 自定义数据集
class ColorDataset(Dataset):
    def __init__(self, image_paths, rgb_values, transform=None):
        self.image_paths = image_paths
        self.rgb_values = rgb_values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
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

# 加载训练集和验证集
train_image_paths = np.load("./dataset/train_image_paths.npy", allow_pickle=True)
train_rgb_values = np.load("./dataset/train_rgb_values.npy", allow_pickle=True)
val_image_paths = np.load("./dataset/val_image_paths.npy", allow_pickle=True)
val_rgb_values = np.load("./dataset/val_rgb_values.npy", allow_pickle=True)
print(f"Training set: {len(train_image_paths)} images")
print(f"Validation set: {len(val_image_paths)} images")

train_dataset = ColorDataset(train_image_paths, train_rgb_values, transform=transform)
val_dataset = ColorDataset(val_image_paths, val_rgb_values, transform=transform)

# 参数设置
batch_size = 16
learning_rate = 0.001
initial_epochs = 2  # 初始训练轮数
additional_epochs = 30  # 继续训练轮数
patience = 3  # 早停机制的耐心值

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的VGG模型
vgg = models.vgg16()
vgg.load_state_dict(torch.load("./model/vgg16-397923af.pth"))

# 修改最后的全连接层
vgg.classifier[6] = nn.Linear(4096, 3)
print(vgg)
# 将模型移动到GPU
vgg.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(vgg.parameters(), lr=learning_rate)

# 创建保存结果的目录
exp_dir = "train_result/exp_0"
counter = 0
while os.path.exists(exp_dir):
    counter += 1
    exp_dir = f"train_result/exp_{counter}"
os.makedirs(exp_dir)

# 保存参数设置
params = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "initial_epochs": initial_epochs,
    "additional_epochs": additional_epochs,
    "patience": patience,
}
with open(os.path.join(exp_dir, "params.json"), "w") as f:
    json.dump(params, f)


# 训练模型的函数
def train_model(
    epochs, exp_dir, start_epoch=0, continue_training=False, prev_exp_dir=None
):
    if continue_training:
        if prev_exp_dir is None:
            raise ValueError(
                "Previous experiment directory must be provided for continuation."
            )
        model_path = os.path.join(prev_exp_dir, "vgg16_color_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No previous training data found at {model_path}")
        vgg.load_state_dict(torch.load(model_path))
        losses_path = os.path.join(prev_exp_dir, "losses.json")
        with open(losses_path, "r") as f:
            losses = json.load(f)
    else:
        losses = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        vgg.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + epochs}"
        )
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{start_epoch + epochs}], Loss: {epoch_loss}")

        # 验证模型
        vgg.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vgg(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        # 检查早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            torch.save(
                vgg.state_dict(), os.path.join(exp_dir, "best_vgg16_color_model.pth")
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    # 保存训练后的模型
    model_path = os.path.join(exp_dir, "vgg16_color_model.pth")
    torch.save(vgg.state_dict(), model_path)

    # 保存损失率
    losses_path = os.path.join(exp_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(losses, f)

    # 绘制并保存损失折线图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-", color="b")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.show()

    print(f"Model trained and saved as {model_path}")
    print(f"Training losses saved as {losses_path}")
    print(f"Loss curve saved as {os.path.join(exp_dir, 'loss_curve.png')}")


# 初始训练
# train_model(initial_epochs, exp_dir)

# 如果需要继续训练，则手动指定上一次的模型路径
prev_exp_dir = "train_result/exp_1"  # 手动指定上一次训练的exp_dir
train_model(
    additional_epochs,
    exp_dir,
    start_epoch=initial_epochs,
    continue_training=True,
    prev_exp_dir=prev_exp_dir,
)
