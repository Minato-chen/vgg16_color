import os
import itertools
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


# 创建保存图像的目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 生成RGB值每隔12的图像
def generate_fixed_color_images(output_dir):
    create_directory(output_dir)
    rgb_values = []
    image_paths = []
    step = 12

    for r, g, b in itertools.product(range(0, 256, step), repeat=3):
        # 生成图像
        image = Image.new("RGB", (64, 64), (r, g, b))
        image_path = os.path.join(output_dir, f"image_{r}_{g}_{b}.png")
        image.save(image_path)

        # 保存图像路径和RGB值
        image_paths.append(image_path)
        rgb_values.append([r, g, b])

    return image_paths, rgb_values


# 划分训练集、验证集和测试集
def split_dataset(image_paths, rgb_values, train_size=0.6, val_size=0.2, test_size=0.2):
    # 首先划分训练集和临时集
    train_paths, temp_paths, train_rgb, temp_rgb = train_test_split(
        image_paths, rgb_values, train_size=train_size, random_state=42
    )
    # 然后在临时集中划分验证集和测试集
    val_test_size = val_size / (val_size + test_size)
    val_paths, test_paths, val_rgb, test_rgb = train_test_split(
        temp_paths, temp_rgb, test_size=val_test_size, random_state=42
    )
    return train_paths, val_paths, test_paths, train_rgb, val_rgb, test_rgb


# 主函数
def main():
    output_dir = "../pure_color_dataset"

    # 生成固定颜色图像数据集
    image_paths, rgb_values = generate_fixed_color_images(output_dir)

    # 划分训练集、验证集和测试集
    train_paths, val_paths, test_paths, train_rgb, val_rgb, test_rgb = split_dataset(
        image_paths, rgb_values, train_size=0.6, val_size=0.2, test_size=0.2
    )

    # 保存训练集、验证集和测试集路径及其对应的RGB值
    np.save("./dataset/train_image_paths.npy", train_paths)
    np.save("./dataset/train_rgb_values.npy", train_rgb)
    np.save("./dataset/val_image_paths.npy", val_paths)
    np.save("./dataset/val_rgb_values.npy", val_rgb)
    np.save("./dataset/test_image_paths.npy", test_paths)
    np.save("./dataset/test_rgb_values.npy", test_rgb)

    print(f"Dataset generated and split into training, validation, and testing sets.")
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Testing set: {len(test_paths)} images")


if __name__ == "__main__":
    main()
