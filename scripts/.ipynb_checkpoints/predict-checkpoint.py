import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import UNet, ResidualUNet


def predict(model_path, image_path, output_path, device='cpu', image_size=512):
    """对单张图像进行滑坡分割预测"""
    # 加载模型
    # model = UNet(n_channels=3, n_classes=1)
    model = ResidualUNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()

    # 后处理
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    prediction_img = Image.fromarray(prediction)
    prediction_img = prediction_img.resize(original_size, Image.NEAREST)

    # 保存结果
    prediction_img.save(output_path)
    print(f"预测结果已保存至: {output_path}")

    return prediction_img


if __name__ == "__main__":
    # 使用示例
    # model_path = "run/bcd_dice_5epoch/best_landslide_unet.pth"
    model_path = "/root/autodl-tmp/recurrence_unet/scripts/run/BFA_5epoch/best_landslide_unet.pth"
    image_path = "test/a1_3-2.jpg"
    output_path = "test/a1_3-2_pre.jpg"

    predict(model_path, image_path, output_path)